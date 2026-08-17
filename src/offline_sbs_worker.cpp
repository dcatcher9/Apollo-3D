/**
 * @file src/offline_sbs_worker.cpp
 * @brief Native scene-lookahead SBS evaluation/conversion worker.
 */

#include "offline_sbs_worker.h"

#include "crypto.h"
#include "depth_coordinate_v2.h"
#include "generated/sbs_adaptive_state_contract.h"
#include "host_sbs_observation_timeline.h"
#include "host_sbs_shader_cache.h"
#include "offline_sbs_contract.h"
#include "offline_scene_planner.h"

#include <algorithm>
#include <array>
#include <bit>
#include <boost/multiprecision/cpp_int.hpp>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#ifdef _WIN32
// clang-format off
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <windows.h>
// clang-format on
#else
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace offline_sbs {
  namespace fs = std::filesystem;
  using namespace std::chrono_literals;

  namespace {
    constexpr std::uint64_t max_sequence = 9999999999ull;
    constexpr std::uintmax_t max_spec_bytes = 4ull * 1024ull * 1024ull;
    constexpr std::uintmax_t max_small_contract_bytes = 64ull * 1024ull * 1024ull;
    constexpr std::uintmax_t max_video_metadata_probe_bytes =
      4ull * 1024ull * 1024ull;
    constexpr std::uintmax_t max_stream_inventory_probe_bytes =
      8ull * 1024ull * 1024ull;
    constexpr std::uintmax_t max_packet_probe_bytes =
      64ull * 1024ull * 1024ull;
    constexpr std::uint64_t max_inventory_streams = 256;
    constexpr std::uint64_t max_inventory_chapters = 65536;
    constexpr std::uint64_t max_auxiliary_packets = 2000000;
    constexpr std::uint64_t max_retained_packet_bytes =
      128ull * 1024ull * 1024ull;
    constexpr std::uint64_t max_retained_timing_bytes =
      128ull * 1024ull * 1024ull;
    constexpr std::uint64_t max_retained_timing_frames =
      max_retained_timing_bytes / sizeof(frame_timing_t);
    static_assert(
      max_retained_timing_frames > 12ull * 60ull * 60ull * 90ull,
      "the timing contract must cover the child timeout at 90 FPS"
    );
    constexpr auto child_poll = 20ms;
    constexpr auto child_timeout = std::chrono::hours(12);
    constexpr auto frame_io_timeout = std::chrono::minutes(5);

    class worker_error: public std::runtime_error {
    public:
      using std::runtime_error::runtime_error;
    };

    std::vector<std::uint64_t> exact_observation_timestamps(
      const media_contract_t &media
    ) {
      using boost::multiprecision::cpp_int;
      if (media.frames.empty() || media.time_base.numerator <= 0 ||
          media.time_base.denominator <= 0) {
        throw worker_error("source observation timeline has no positive time base/frames");
      }
      const cpp_int first_pts = media.frames.front().pts;
      const cpp_int numerator = media.time_base.numerator;
      const cpp_int denominator = media.time_base.denominator;
      const cpp_int maximum = std::numeric_limits<std::uint64_t>::max();
      std::vector<std::uint64_t> timestamps;
      timestamps.reserve(media.frames.size());
      for (std::size_t index = 0u; index < media.frames.size(); ++index) {
        const auto &frame = media.frames[index];
        if (frame.sequence != index + 1u) {
          throw worker_error("source observation timeline sequence is not contiguous");
        }
        const cpp_int delta = cpp_int(frame.pts) - first_pts;
        const cpp_int elapsed_us = delta * numerator * 1000000 / denominator;
        if (elapsed_us < 0 || elapsed_us >= maximum) {
          throw worker_error("source observation timestamp is outside uint64 microseconds");
        }
        timestamps.push_back((elapsed_us + 1).convert_to<std::uint64_t>());
      }
      if (!models::host_sbs_observation_timeline::valid_timestamps(timestamps)) {
        throw worker_error("source observation timestamps are zero or regressed");
      }
      return timestamps;
    }

    void write_observation_timeline(
      const fs::path &path,
      const media_contract_t &media
    ) {
      const auto timestamps = exact_observation_timestamps(media);
      std::string error;
      if (!models::host_sbs_observation_timeline::write(path, timestamps, error)) {
        throw worker_error(error);
      }
    }

    constexpr bool can_retain_another_timing_frame(
      const std::uint64_t retained_frames
    ) {
      return retained_frames < max_retained_timing_frames;
    }

    constexpr bool can_retain_auxiliary_packets(
      const std::uint64_t retained_packets,
      const std::uint64_t additional_packets
    ) {
      return retained_packets <= max_auxiliary_packets &&
             additional_packets <=
               max_auxiliary_packets - retained_packets;
    }

    constexpr bool can_consume_packet_probe_bytes(
      const std::uintmax_t consumed_bytes,
      const std::uintmax_t additional_bytes
    ) {
      return consumed_bytes <= max_packet_probe_bytes &&
             additional_bytes <= max_packet_probe_bytes - consumed_bytes;
    }

    std::string secure_frame_bridge_token() {
      std::array<unsigned char, 32> random {};
      if (RAND_bytes(random.data(), static_cast<int>(random.size())) != 1) {
        throw worker_error("cannot generate the loopback frame bridge capability");
      }
      static constexpr char hex[] = "0123456789abcdef";
      std::string token(random.size() * 2, '\0');
      for (std::size_t index = 0; index < random.size(); ++index) {
        token[index * 2] = hex[random[index] >> 4];
        token[index * 2 + 1] = hex[random[index] & 0x0f];
      }
      return token;
    }

    std::string path_utf8(const fs::path &path) {
#ifdef _WIN32
      const auto wide = path.wstring();
      if (wide.empty()) {
        return {};
      }
      const int size = WideCharToMultiByte(
        CP_UTF8,
        WC_ERR_INVALID_CHARS,
        wide.data(),
        static_cast<int>(wide.size()),
        nullptr,
        0,
        nullptr,
        nullptr
      );
      if (size <= 0) {
        throw worker_error("cannot encode a filesystem path as UTF-8");
      }
      std::string result(static_cast<std::size_t>(size), '\0');
      if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide.data(), static_cast<int>(wide.size()), result.data(), size, nullptr, nullptr) != size) {
        throw worker_error("cannot encode a filesystem path as UTF-8");
      }
      return result;
#else
      return path.string();
#endif
    }

    fs::path path_from_utf8(const std::string &value) {
#ifdef _WIN32
      if (value.empty()) {
        return {};
      }
      const int size = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        nullptr,
        0
      );
      if (size <= 0) {
        throw worker_error("worker contract contains invalid UTF-8");
      }
      std::wstring wide(static_cast<std::size_t>(size), L'\0');
      if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), static_cast<int>(value.size()), wide.data(), size) != size) {
        throw worker_error("worker contract contains invalid UTF-8");
      }
      return fs::path(wide);
#else
      return fs::path(value);
#endif
    }

    std::string lower(std::string value) {
      std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](const unsigned char character) {
          return static_cast<char>(std::tolower(character));
        }
      );
      return value;
    }

    std::string sha256_hex(const std::string_view bytes) {
      static constexpr char digits[] = "0123456789abcdef";
      const auto digest = crypto::hash(bytes);
      std::string result(digest.size() * 2, '\0');
      for (std::size_t index = 0; index < digest.size(); ++index) {
        result[index * 2] = digits[digest[index] >> 4u];
        result[index * 2 + 1] = digits[digest[index] & 0x0fu];
      }
      return result;
    }

    bool valid_sha256_hex(const std::string_view value) {
      return value.size() == 64 &&
             std::all_of(value.begin(), value.end(), [](const unsigned char character) {
               return
                 (character >= '0' && character <= '9') ||
                 (character >= 'a' && character <= 'f');
             });
    }

    bool constant_time_equal(
      const std::string_view left,
      const std::string_view right
    ) {
      if (left.size() != right.size()) {
        return false;
      }
      unsigned char difference = 0;
      for (std::size_t index = 0; index < left.size(); ++index) {
        difference |= static_cast<unsigned char>(left[index] ^ right[index]);
      }
      return difference == 0;
    }

    std::string read_bounded_bytes(
      const fs::path &path,
      const std::uintmax_t max_bytes
    ) {
      std::error_code ec;
      if (!fs::is_regular_file(path, ec) || ec) {
        throw worker_error("worker specification is missing");
      }
      const auto reported_bytes = fs::file_size(path, ec);
      if (ec || reported_bytes == 0 || reported_bytes > max_bytes) {
        throw worker_error("worker specification has an invalid size");
      }
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        throw worker_error("cannot open worker specification");
      }
      std::string bytes(
        static_cast<std::size_t>(max_bytes) + 1,
        '\0'
      );
      stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
      const auto read = stream.gcount();
      if (read <= 0 || static_cast<std::uintmax_t>(read) > max_bytes) {
        throw worker_error("worker specification changed to an invalid size");
      }
      if (stream.bad()) {
        throw worker_error("cannot read worker specification");
      }
      bytes.resize(static_cast<std::size_t>(read));
      return bytes;
    }

    bool unknown_metadata(const nlohmann::json &value) {
      if (value.is_null()) {
        return true;
      }
      if (!value.is_string()) {
        return false;
      }
      const auto text = lower(value.get<std::string>());
      return text.empty() || text == "unknown" || text == "reserved" ||
             text == "unspecified" || text == "n/a";
    }

    const nlohmann::json *object_member(
      const nlohmann::json &object,
      const std::string_view name
    ) {
      if (!object.is_object()) {
        return nullptr;
      }
      const auto found = object.find(name);
      return found == object.end() ? nullptr : &*found;
    }

    std::string required_string(
      const nlohmann::json &object,
      const std::string_view name
    ) {
      const auto *value = object_member(object, name);
      if (!value || !value->is_string()) {
        throw worker_error("worker contract field '" + std::string(name) + "' must be a string");
      }
      const auto result = value->get<std::string>();
      if (result.empty()) {
        throw worker_error("worker contract field '" + std::string(name) + "' must not be empty");
      }
      return result;
    }

    fs::path required_absolute_path(
      const nlohmann::json &object,
      const std::string_view name
    ) {
      auto path = path_from_utf8(required_string(object, name));
      if (!path.is_absolute()) {
        throw worker_error("worker contract path '" + std::string(name) + "' must be absolute");
      }
      return path.lexically_normal();
    }

    template<class Input>
    nlohmann::json parse_json_without_duplicate_keys(Input &&input) {
      std::map<int, std::set<std::string>> object_keys;
      const auto callback = [&object_keys](
                              const int depth,
                              const nlohmann::json::parse_event_t event,
                              nlohmann::json &parsed) {
        if (event == nlohmann::json::parse_event_t::object_start) {
          object_keys[depth].clear();
        } else if (event == nlohmann::json::parse_event_t::key) {
          if (depth <= 0 || !parsed.is_string()) {
            throw worker_error("JSON object key has invalid parser depth or type");
          }
          auto owner = object_keys.find(depth - 1);
          if (owner == object_keys.end()) {
            throw worker_error("JSON object key has no owning object");
          }
          const auto &key = parsed.get_ref<const std::string &>();
          if (!owner->second.insert(key).second) {
            throw worker_error("JSON object contains duplicate key '" + key + "'");
          }
        } else if (event == nlohmann::json::parse_event_t::object_end) {
          object_keys.erase(depth);
        }
        return true;
      };
      return nlohmann::json::parse(
        std::forward<Input>(input), callback, true, false
      );
    }

    nlohmann::json read_json(
      const fs::path &path,
      const std::uintmax_t max_bytes = max_small_contract_bytes
    ) {
      std::error_code ec;
      if (!fs::is_regular_file(path, ec) || ec) {
        throw worker_error("JSON contract is missing: " + path_utf8(path));
      }
      const auto bytes = fs::file_size(path, ec);
      if (ec || bytes == 0 || bytes > max_bytes) {
        throw worker_error("JSON contract has an invalid size: " + path_utf8(path));
      }
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        throw worker_error("cannot open JSON contract: " + path_utf8(path));
      }
      try {
        return parse_json_without_duplicate_keys(stream);
      } catch (const std::exception &exception) {
        throw worker_error(
          "cannot parse JSON contract " + path_utf8(path) + ": " +
          exception.what()
        );
      }
    }

#ifdef _WIN32
    bool write_all(HANDLE handle, const void *data, std::size_t size) {
      const auto *bytes = static_cast<const std::uint8_t *>(data);
      std::size_t offset = 0;
      while (offset < size) {
        const DWORD chunk = static_cast<DWORD>(
          std::min<std::size_t>(
            size - offset,
            std::numeric_limits<DWORD>::max()
          )
        );
        DWORD written = 0;
        if (!WriteFile(handle, bytes + offset, chunk, &written, nullptr) || written != chunk) {
          return false;
        }
        offset += written;
      }
      return true;
    }
#endif

    void write_bytes_atomic(
      const fs::path &path,
      const std::vector<std::uint8_t> &bytes
    ) {
      std::error_code ec;
      fs::create_directories(path.parent_path(), ec);
      if (ec) {
        throw worker_error("cannot create contract directory: " + ec.message());
      }
      auto temporary = path;
      temporary += L".tmp";
#ifdef _WIN32
      const HANDLE handle = CreateFileW(
        temporary.c_str(),
        GENERIC_WRITE,
        0,
        nullptr,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH,
        nullptr
      );
      if (handle == INVALID_HANDLE_VALUE) {
        throw worker_error("cannot create atomic temporary file: " + path_utf8(temporary));
      }
      bool ok =
        write_all(handle, bytes.data(), bytes.size()) &&
        FlushFileBuffers(handle);
      const bool closed = CloseHandle(handle);
      ok = ok && closed;
      if (!ok) {
        fs::remove(temporary, ec);
        throw worker_error("cannot durably write: " + path_utf8(path));
      }
      if (!MoveFileExW(
            temporary.c_str(),
            path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH
          )) {
        fs::remove(temporary, ec);
        throw worker_error("cannot atomically publish: " + path_utf8(path));
      }
#else
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      output.write(
        reinterpret_cast<const char *>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
      );
      output.close();
      if (!output) {
        fs::remove(temporary, ec);
        throw worker_error("cannot write: " + path.string());
      }
      fs::rename(temporary, path, ec);
      if (ec) {
        throw worker_error("cannot publish: " + ec.message());
      }
#endif
    }

    void write_json_atomic(const fs::path &path, const nlohmann::json &value) {
      const auto text = value.dump(2) + "\n";
      write_bytes_atomic(
        path,
        std::vector<std::uint8_t>(text.begin(), text.end())
      );
    }

    void write_json_atomic_bounded(
      const fs::path &path,
      const nlohmann::json &value,
      const std::uintmax_t max_bytes,
      const std::string_view description
    ) {
      const auto text = value.dump(2) + "\n";
      if (text.size() > max_bytes) {
        throw worker_error(
          std::string {description} + " exceeds its " +
          std::to_string(max_bytes) + "-byte serialized contract"
        );
      }
      write_bytes_atomic(
        path,
        std::vector<std::uint8_t>(text.begin(), text.end())
      );
    }

    void remove_file_checked(const fs::path &path) {
      std::error_code ec;
      if (!fs::remove(path, ec) || ec) {
        throw worker_error("cannot release temporary file: " + path_utf8(path));
      }
    }

    void remove_file_if_present_checked(const fs::path &path) {
      std::error_code ec;
      const bool exists = fs::exists(path, ec);
      if (ec) {
        throw worker_error(
          "cannot inspect temporary file: " + path_utf8(path)
        );
      }
      if (exists) {
        remove_file_checked(path);
      }
    }

    void remove_empty_directory_checked(const fs::path &path) {
      std::error_code ec;
      if (!fs::remove(path, ec) || ec) {
        throw worker_error(
          "cannot release empty temporary directory: " + path_utf8(path)
        );
      }
    }

    /**
     * Atomically reserves the manager-selected staging pathname and keeps the exact
     * file identity open without delete sharing while FFmpeg writes it. Failed output
     * is deliberately preserved: cleanup never performs a pathname deletion that
     * could target a user file after the child loses its identity lock.
     */
    class staging_output_claim_t {
    public:
      explicit staging_output_claim_t(fs::path path):
          path_(std::move(path)) {
#ifdef _WIN32
        handle_ = CreateFileW(
          path_.c_str(),
          FILE_READ_ATTRIBUTES,
          FILE_SHARE_READ | FILE_SHARE_WRITE,
          nullptr,
          CREATE_NEW,
          FILE_ATTRIBUTE_NORMAL,
          nullptr
        );
        if (handle_ == INVALID_HANDLE_VALUE) {
          throw worker_error(
            "cannot exclusively reserve conversion staging output"
          );
        }
        BY_HANDLE_FILE_INFORMATION information {};
        if (
          !GetFileInformationByHandle(handle_, &information) ||
          (information.dwFileAttributes &
             (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0
        ) {
          cleanup_owned();
          throw worker_error("cannot attest reserved staging output");
        }
        volume_serial_ = information.dwVolumeSerialNumber;
        file_index_high_ = information.nFileIndexHigh;
        file_index_low_ = information.nFileIndexLow;
#else
        descriptor_ = ::open(
          path_.c_str(),
          O_CREAT | O_EXCL | O_RDWR | O_CLOEXEC,
          0600
        );
        struct stat information {};
        if (
          descriptor_ < 0 ||
          ::fstat(descriptor_, &information) != 0 ||
          !S_ISREG(information.st_mode)
        ) {
          if (descriptor_ >= 0) {
            ::close(descriptor_);
            descriptor_ = -1;
          }
          throw worker_error(
            "cannot exclusively reserve conversion staging output"
          );
        }
        device_ = information.st_dev;
        inode_ = information.st_ino;
#endif
      }

      ~staging_output_claim_t() {
        cleanup_owned();
      }

      staging_output_claim_t(const staging_output_claim_t &) = delete;
      staging_output_claim_t &operator=(const staging_output_claim_t &) = delete;

      staging_output_claim_t(staging_output_claim_t &&other) noexcept:
          path_(std::move(other.path_))
#ifdef _WIN32
          ,
          handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)),
          volume_serial_(other.volume_serial_),
          file_index_high_(other.file_index_high_),
          file_index_low_(other.file_index_low_)
#else
          ,
          descriptor_(std::exchange(other.descriptor_, -1)),
          device_(other.device_),
          inode_(other.inode_)
#endif
      {
      }

      staging_output_claim_t &operator=(
        staging_output_claim_t &&other
      ) noexcept {
        if (this == &other) {
          return *this;
        }
        cleanup_owned();
        path_ = std::move(other.path_);
#ifdef _WIN32
        handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
        volume_serial_ = other.volume_serial_;
        file_index_high_ = other.file_index_high_;
        file_index_low_ = other.file_index_low_;
#else
        descriptor_ = std::exchange(other.descriptor_, -1);
        device_ = other.device_;
        inode_ = other.inode_;
#endif
        return *this;
      }

      [[nodiscard]] nlohmann::json identity_json() const {
#ifdef _WIN32
        const auto information = verified_path_information();
        const auto size =
          (static_cast<std::uint64_t>(information.nFileSizeHigh) << 32u) |
          information.nFileSizeLow;
        if (size == 0) {
          throw worker_error("conversion staging output is empty");
        }
        return {
          {"volume_serial", information.dwVolumeSerialNumber},
          {"file_index_high", information.nFileIndexHigh},
          {"file_index_low", information.nFileIndexLow},
          {"size_bytes", size},
        };
#else
        struct stat information {};
        if (
          descriptor_ < 0 ||
          ::fstat(descriptor_, &information) != 0 ||
          information.st_size <= 0 ||
          !path_matches_claim()
        ) {
          throw worker_error(
            "conversion staging output identity changed or is empty"
          );
        }
        return {
          {"device", static_cast<std::uint64_t>(information.st_dev)},
          {"inode", static_cast<std::uint64_t>(information.st_ino)},
          {"size_bytes", static_cast<std::uint64_t>(information.st_size)},
        };
#endif
      }

      void release_for_publish() {
        (void) identity_json();
#ifdef _WIN32
        CloseHandle(std::exchange(handle_, INVALID_HANDLE_VALUE));
#else
        ::close(std::exchange(descriptor_, -1));
#endif
      }

    private:
#ifdef _WIN32
      [[nodiscard]] BY_HANDLE_FILE_INFORMATION
      verified_path_information() const {
        if (handle_ == INVALID_HANDLE_VALUE) {
          throw worker_error("conversion staging output claim is not active");
        }
        const HANDLE observed = CreateFileW(
          path_.c_str(),
          FILE_READ_ATTRIBUTES,
          FILE_SHARE_READ | FILE_SHARE_WRITE,
          nullptr,
          OPEN_EXISTING,
          FILE_FLAG_OPEN_REPARSE_POINT,
          nullptr
        );
        if (observed == INVALID_HANDLE_VALUE) {
          throw worker_error(
            "conversion staging output identity changed"
          );
        }
        BY_HANDLE_FILE_INFORMATION information {};
        const bool valid =
          GetFileInformationByHandle(observed, &information) &&
          (information.dwFileAttributes &
             (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) == 0 &&
          information.dwVolumeSerialNumber == volume_serial_ &&
          information.nFileIndexHigh == file_index_high_ &&
          information.nFileIndexLow == file_index_low_;
        CloseHandle(observed);
        if (!valid) {
          throw worker_error(
            "conversion staging output identity changed"
          );
        }
        return information;
      }

      void cleanup_owned() noexcept {
        if (handle_ == INVALID_HANDLE_VALUE) {
          return;
        }
        // DELETE access is intentionally not held: the packaged FFmpeg does not share
        // delete access when opening an existing output. Keeping a no-delete read-
        // attribute handle locks the pathname throughout conversion. On failure, close
        // and preserve the exact claimed file; never fall back to a racy path deletion.
        CloseHandle(std::exchange(handle_, INVALID_HANDLE_VALUE));
      }
#else
      [[nodiscard]] bool path_matches_claim() const noexcept {
        struct stat information {};
        return
          ::lstat(path_.c_str(), &information) == 0 &&
          S_ISREG(information.st_mode) &&
          information.st_dev == device_ &&
          information.st_ino == inode_;
      }

      void cleanup_owned() noexcept {
        if (descriptor_ < 0) {
          return;
        }
        // A portable lstat()+unlink() sequence has a pathname race. Preserve the file
        // rather than ever deleting an identity that was not created by this run.
        ::close(std::exchange(descriptor_, -1));
      }
#endif

      fs::path path_;
#ifdef _WIN32
      HANDLE handle_ = INVALID_HANDLE_VALUE;
      DWORD volume_serial_ = 0;
      DWORD file_index_high_ = 0;
      DWORD file_index_low_ = 0;
#else
      int descriptor_ = -1;
      dev_t device_ {};
      ino_t inode_ {};
#endif
    };

    bool path_is_within(
      const fs::path &candidate,
      const fs::path &parent
    ) {
      const auto normalized_candidate = candidate.lexically_normal();
      const auto normalized_parent = parent.lexically_normal();
      auto candidate_part = normalized_candidate.begin();
      for (auto parent_part = normalized_parent.begin();
           parent_part != normalized_parent.end();
           ++parent_part, ++candidate_part) {
        if (candidate_part == normalized_candidate.end()) {
          return false;
        }
#ifdef _WIN32
        if (lower(path_utf8(*candidate_part)) != lower(path_utf8(*parent_part))) {
#else
        if (*candidate_part != *parent_part) {
#endif
          return false;
        }
      }
      return true;
    }

    std::string frame_id(const std::uint64_t sequence) {
      char buffer[16] {};
      std::snprintf(buffer, sizeof(buffer), "%010llu", static_cast<unsigned long long>(sequence));
      return buffer;
    }

    std::int64_t parse_integer_json(
      const nlohmann::json &value,
      const std::string_view description
    ) {
      if (value.is_number_integer()) {
        return value.get<std::int64_t>();
      }
      if (!value.is_string()) {
        throw worker_error(
          "ffprobe " + std::string(description) + " is not an integer"
        );
      }
      const auto text = value.get<std::string>();
      std::int64_t result = 0;
      const auto parsed = std::from_chars(
        text.data(),
        text.data() + text.size(),
        result
      );
      if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size()) {
        throw worker_error(
          "ffprobe " + std::string(description) + " is not an integer"
        );
      }
      return result;
    }

    std::optional<std::int64_t> optional_integer_json(
      const nlohmann::json &object,
      const std::string_view name
    ) {
      const auto *value = object_member(object, name);
      if (!value || value->is_null() || (value->is_string() && lower(value->get<std::string>()) == "n/a")) {
        return std::nullopt;
      }
      return parse_integer_json(*value, name);
    }

    std::optional<long double> parse_decimal_seconds(
      const nlohmann::json &value
    ) {
      if (!value.is_string() && !value.is_number()) {
        return std::nullopt;
      }
      try {
        const auto text = value.is_string() ?
                            value.get<std::string>() :
                            value.dump();
        std::size_t consumed = 0;
        const auto seconds = std::stold(text, &consumed);
        if (consumed != text.size() || !std::isfinite(seconds) || seconds <= 0.0L) {
          return std::nullopt;
        }
        return seconds;
      } catch (const std::exception &) {
        return std::nullopt;
      }
    }

    std::optional<long double> parse_clock_duration(
      const nlohmann::json &value
    ) {
      if (!value.is_string()) {
        return std::nullopt;
      }
      const auto text = value.get<std::string>();
      static const std::regex pattern {
        R"(^([0-9]+):([0-5][0-9]):([0-5][0-9](?:\.[0-9]+)?)$)"
      };
      std::smatch match;
      if (!std::regex_match(text, match, pattern)) {
        return std::nullopt;
      }
      try {
        return static_cast<long double>(std::stoull(match[1].str())) * 3600.0L +
               static_cast<long double>(std::stoull(match[2].str())) * 60.0L +
               std::stold(match[3].str());
      } catch (const std::exception &) {
        return std::nullopt;
      }
    }

    std::optional<std::int64_t> stream_duration_ticks(
      const nlohmann::json &stream,
      const rational_t time_base
    ) {
      if (const auto duration = optional_integer_json(stream, "duration_ts"); duration && *duration > 0) {
        return duration;
      }
      std::optional<long double> duration_seconds;
      if (const auto *duration = object_member(stream, "duration")) {
        duration_seconds = parse_decimal_seconds(*duration);
      }
      if (!duration_seconds) {
        if (const auto *tags = object_member(stream, "tags"); tags && tags->is_object()) {
          if (const auto *duration = object_member(*tags, "DURATION")) {
            duration_seconds = parse_clock_duration(*duration);
          }
        }
      }
      if (!duration_seconds) {
        return std::nullopt;
      }
      const long double tick_seconds =
        static_cast<long double>(time_base.numerator) /
        time_base.denominator;
      const long double exact_ticks = *duration_seconds / tick_seconds;
      const auto rounded = std::llround(exact_ticks);
      if (rounded <= 0 || std::abs(exact_ticks - rounded) > 0.5L + 1e-9L) {
        return std::nullopt;
      }
      return rounded;
    }

    std::int64_t gcd_abs(std::int64_t left, std::int64_t right) {
      left = std::abs(left);
      right = std::abs(right);
      while (right != 0) {
        const auto remainder = left % right;
        left = right;
        right = remainder;
      }
      return left;
    }

    std::int64_t nonnegative_tick_difference(
      std::int64_t later,
      std::int64_t earlier,
      std::string_view description
    );

    std::int64_t positive_tick_difference(
      std::int64_t later,
      std::int64_t earlier,
      std::string_view description
    );

    std::uint64_t checked_byte_sum(
      const std::uint64_t left,
      const std::uint64_t right,
      const std::string_view description
    ) {
      if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        throw worker_error(std::string(description) + " byte count overflows");
      }
      return left + right;
    }

    bool rational_ticks_within_actual_ticks(
      const std::int64_t source_ticks,
      const rational_t source_time_base,
      const std::int64_t actual_ticks,
      const rational_t actual_time_base,
      const std::uint64_t tolerance_ticks
    ) {
      using boost::multiprecision::cpp_int;
      if (source_time_base.numerator <= 0 || source_time_base.denominator <= 0 || actual_time_base.numerator <= 0 || actual_time_base.denominator <= 0) {
        throw worker_error("cannot compare an invalid media time base");
      }
      const cpp_int delta = boost::multiprecision::abs(
        cpp_int(source_ticks) * source_time_base.numerator *
          actual_time_base.denominator -
        cpp_int(actual_ticks) * actual_time_base.numerator *
          source_time_base.denominator
      );
      const cpp_int allowed =
        cpp_int(tolerance_ticks) * actual_time_base.numerator *
        source_time_base.denominator;
      return delta <= allowed;
    }

    void require_millisecond_or_finer(
      const rational_t time_base,
      const std::string_view description
    ) {
      using boost::multiprecision::cpp_int;
      if (time_base.numerator <= 0 || time_base.denominator <= 0 || cpp_int(time_base.numerator) * 1000 > time_base.denominator) {
        throw worker_error(
          std::string(description) +
          " time base is coarser than the 1 ms Matroska contract"
        );
      }
    }

    void validate_timeline_contract(
      const media_contract_t &source,
      const media_contract_t &actual,
      const std::uint64_t tolerance_actual_ticks
    ) {
      if (source.frames.size() != actual.frames.size()) {
        throw worker_error(
          "encoded frame count differs from the source timeline"
        );
      }
      for (std::size_t index = 0; index < source.frames.size(); ++index) {
        const auto &wanted = source.frames[index];
        const auto &got = actual.frames[index];
        if (!rational_ticks_within_actual_ticks(wanted.pts, source.time_base, got.pts, actual.time_base, tolerance_actual_ticks) || !rational_ticks_within_actual_ticks(wanted.duration, source.time_base, got.duration, actual.time_base, tolerance_actual_ticks)) {
          throw worker_error(
            "encoded presentation timeline differs at frame " +
            std::to_string(index + 1)
          );
        }
      }
      const auto wanted_total = positive_tick_difference(
        source.end_pts_exclusive(),
        source.first_pts(),
        "source presentation duration"
      );
      const auto actual_total = positive_tick_difference(
        actual.end_pts_exclusive(),
        actual.first_pts(),
        "encoded presentation duration"
      );
      if (!rational_ticks_within_actual_ticks(
            wanted_total,
            source.time_base,
            actual_total,
            actual.time_base,
            tolerance_actual_ticks
          )) {
        throw worker_error(
          "encoded presentation duration differs from source"
        );
      }
    }

    long double rational_error_in_actual_ticks(
      const std::int64_t source_ticks,
      const rational_t source_time_base,
      const std::int64_t actual_ticks,
      const rational_t actual_time_base
    ) {
      using boost::multiprecision::cpp_int;
      const cpp_int delta = boost::multiprecision::abs(
        cpp_int(source_ticks) * source_time_base.numerator *
          actual_time_base.denominator -
        cpp_int(actual_ticks) * actual_time_base.numerator *
          source_time_base.denominator
      );
      const cpp_int denominator =
        cpp_int(source_time_base.denominator) *
        actual_time_base.numerator;
      if (denominator <= 0) {
        throw worker_error("cannot report an invalid timeline time base");
      }
      return delta.convert_to<long double>() /
             denominator.convert_to<long double>();
    }

    nlohmann::json timeline_error_report(
      const media_contract_t &source,
      const media_contract_t &actual
    ) {
      if (source.frames.size() != actual.frames.size()) {
        throw worker_error("cannot report a mismatched media timeline");
      }
      long double max_pts_error = 0;
      long double max_duration_error = 0;
      for (std::size_t index = 0; index < source.frames.size(); ++index) {
        max_pts_error = std::max(
          max_pts_error,
          rational_error_in_actual_ticks(
            source.frames[index].pts,
            source.time_base,
            actual.frames[index].pts,
            actual.time_base
          )
        );
        max_duration_error = std::max(
          max_duration_error,
          rational_error_in_actual_ticks(
            source.frames[index].duration,
            source.time_base,
            actual.frames[index].duration,
            actual.time_base
          )
        );
      }
      const auto source_duration = positive_tick_difference(
        source.end_pts_exclusive(),
        source.first_pts(),
        "source report duration"
      );
      const auto actual_duration = positive_tick_difference(
        actual.end_pts_exclusive(),
        actual.first_pts(),
        "output report duration"
      );
      const auto total_error = rational_error_in_actual_ticks(
        source_duration,
        source.time_base,
        actual_duration,
        actual.time_base
      );
      return {
        {"source_time_base", {
                               {"numerator", source.time_base.numerator},
                               {"denominator", source.time_base.denominator},
                             }},
        {"output_time_base", {
                               {"numerator", actual.time_base.numerator},
                               {"denominator", actual.time_base.denominator},
                             }},
        {"max_pts_error_output_ticks", static_cast<double>(max_pts_error)},
        {"max_duration_error_output_ticks", static_cast<double>(max_duration_error)},
        {"end_to_end_duration_error_output_ticks", static_cast<double>(total_error)},
        {"exact", max_pts_error == 0 && max_duration_error == 0 && total_error == 0},
      };
    }

    std::int64_t nonnegative_tick_difference(
      const std::int64_t later,
      const std::int64_t earlier,
      const std::string_view description
    ) {
      using boost::multiprecision::cpp_int;
      const cpp_int difference = cpp_int(later) - earlier;
      if (difference < 0 || difference > std::numeric_limits<std::int64_t>::max()) {
        throw worker_error(
          std::string(description) + " is not a nonnegative int64 interval"
        );
      }
      return difference.convert_to<std::int64_t>();
    }

    std::int64_t positive_tick_difference(
      const std::int64_t later,
      const std::int64_t earlier,
      const std::string_view description
    ) {
      const auto difference =
        nonnegative_tick_difference(later, earlier, description);
      if (difference == 0) {
        throw worker_error(
          std::string(description) + " is not a positive int64 interval"
        );
      }
      return difference;
    }

    rational_t parse_rational(
      const nlohmann::json &value,
      const std::string_view description
    ) {
      if (!value.is_string()) {
        throw worker_error(
          "ffprobe " + std::string(description) + " is missing"
        );
      }
      const auto text = value.get<std::string>();
      const auto split = text.find('/');
      if (split == std::string::npos) {
        throw worker_error(
          "ffprobe " + std::string(description) + " is not rational"
        );
      }
      std::int64_t numerator = 0;
      std::int64_t denominator = 0;
      const auto first = std::from_chars(
        text.data(),
        text.data() + split,
        numerator
      );
      const auto second = std::from_chars(
        text.data() + split + 1,
        text.data() + text.size(),
        denominator
      );
      if (first.ec != std::errc {} || first.ptr != text.data() + split || second.ec != std::errc {} || second.ptr != text.data() + text.size() || numerator <= 0 || denominator <= 0) {
        throw worker_error(
          "ffprobe " + std::string(description) +
          " must be a positive rational"
        );
      }
      const auto divisor = gcd_abs(numerator, denominator);
      return {numerator / divisor, denominator / divisor};
    }

    bool looks_high_bit_depth(const std::string &format) {
      static const std::regex pattern {
        R"((?:p|gbrp|gray|rgb|bgr)(?:9|10|12|14|16)(?:le|be)?(?:$|[^0-9]))",
        std::regex::icase
      };
      return std::regex_search(format, pattern) ||
             lower(format).find("p010") != std::string::npos ||
             lower(format).find("p012") != std::string::npos ||
             lower(format).find("p016") != std::string::npos;
    }

    bool has_alpha_channel(const std::string &format) {
      const auto value = lower(format);
      return
        value.starts_with("rgba") ||
        value.starts_with("bgra") ||
        value.starts_with("argb") ||
        value.starts_with("abgr") ||
        value.starts_with("yuva") ||
        value.starts_with("gbrap") ||
        value == "ya8" ||
        value.starts_with("ya16") ||
        value.starts_with("ayuv") ||
        value == "vuya" ||
        value == "uyva" ||
        value == "pal8";
    }

    std::string side_data_type(const nlohmann::json &item) {
      if (!item.is_object()) {
        return {};
      }
      const auto *type = object_member(item, "side_data_type");
      return type && type->is_string() ? type->get<std::string>() : std::string {};
    }

    bool dynamic_hdr_side_data(const std::string &type) {
      const auto text = lower(type);
      return text.find("dovi") != std::string::npos ||
             text.find("dolby vision") != std::string::npos ||
             text.find("hdr10+") != std::string::npos ||
             text.find("smpte2094") != std::string::npos ||
             text.find("dynamic hdr") != std::string::npos ||
             text.find("hdr dynamic") != std::string::npos;
    }

    bool supported_static_video_side_data(const std::string &type) {
      const auto text = lower(type);
      return text == "mastering display metadata" ||
             text == "content light level metadata";
    }

    bool encoder_generated_side_data(const std::string &type) {
      const auto text = lower(type);
      return text.find("user data unregistered") != std::string::npos ||
             text == "video encoding parameters";
    }

    nlohmann::json normalize_static_side_data(nlohmann::json item) {
      static const std::regex rational_pattern {R"(^-?\d+/-?\d+$)"};
      for (auto &[key, value] : item.items()) {
        if (!value.is_string()) {
          continue;
        }
        const auto text = value.get<std::string>();
        if (!std::regex_match(text, rational_pattern)) {
          continue;
        }
        const auto split = text.find('/');
        std::int64_t numerator = 0;
        std::int64_t denominator = 0;
        const auto first = std::from_chars(
          text.data(),
          text.data() + split,
          numerator
        );
        const auto second = std::from_chars(
          text.data() + split + 1,
          text.data() + text.size(),
          denominator
        );
        if (first.ec != std::errc {} || second.ec != std::errc {} || denominator == 0) {
          continue;
        }
        if (denominator < 0) {
          numerator = -numerator;
          denominator = -denominator;
        }
        const auto divisor = gcd_abs(numerator, denominator);
        value = std::to_string(numerator / divisor) + "/" +
                std::to_string(denominator / divisor);
      }
      return item;
    }

    void merge_canonical_side_data(
      const nlohmann::json &owner,
      const std::string_view wanted,
      std::optional<nlohmann::json> &canonical
    ) {
      const auto *list = object_member(owner, "side_data_list");
      if (!list || !list->is_array()) {
        return;
      }
      for (const auto &item : *list) {
        if (side_data_type(item) != wanted) {
          continue;
        }
        auto normalized = item;
        normalized.erase("side_data_type");
        normalized = normalize_static_side_data(std::move(normalized));
        if (!canonical) {
          canonical = std::move(normalized);
        } else if (*canonical != normalized) {
          throw worker_error(
            "ffprobe reports changing/conflicting static " +
            std::string(wanted)
          );
        }
      }
    }

    std::string color_property(
      const nlohmann::json &stream,
      const nlohmann::json &first_frame,
      const std::string_view name
    ) {
      const auto *stream_value = object_member(stream, name);
      const auto *frame_value = object_member(first_frame, name);
      const bool stream_known = stream_value && !unknown_metadata(*stream_value);
      const bool frame_known = frame_value && !unknown_metadata(*frame_value);
      if (stream_known && frame_known && *stream_value != *frame_value) {
        throw worker_error(
          "ffprobe stream/frame " + std::string(name) + " mismatch"
        );
      }
      const auto *value = stream_known ? stream_value :
                                         (frame_known ? frame_value : nullptr);
      if (!value) {
        return {};
      }
      if (!value->is_string()) {
        throw worker_error(
          "ffprobe " + std::string(name) + " is not a string"
        );
      }
      return value->get<std::string>();
    }

    void reject_rotation(const nlohmann::json &stream) {
      std::vector<double> rotations;
      const auto *tags = object_member(stream, "tags");
      if (tags && tags->is_object()) {
        const auto *rotate = object_member(*tags, "rotate");
        if (rotate) {
          try {
            rotations.push_back(
              rotate->is_string() ?
                std::stod(rotate->get<std::string>()) :
                rotate->get<double>()
            );
          } catch (const std::exception &) {
            throw worker_error("ffprobe rotation metadata is malformed");
          }
        }
      }
      const auto *side_data = object_member(stream, "side_data_list");
      if (side_data && side_data->is_array()) {
        for (const auto &item : *side_data) {
          const auto *rotation = object_member(item, "rotation");
          if (!rotation) {
            continue;
          }
          try {
            rotations.push_back(
              rotation->is_string() ?
                std::stod(rotation->get<std::string>()) :
                rotation->get<double>()
            );
          } catch (const std::exception &) {
            throw worker_error("ffprobe rotation side data is malformed");
          }
        }
      }
      for (const auto rotation : rotations) {
        if (!std::isfinite(rotation) || std::abs(std::remainder(rotation, 360.0)) > 1e-6) {
          throw worker_error(
            "rotated video is unsupported; select an already-oriented resolution"
          );
        }
      }
    }

    std::string zscale_range(const std::string &range) {
      const auto value = lower(range);
      if (value == "tv" || value == "mpeg" || value == "limited") {
        return "limited";
      }
      if (value == "pc" || value == "jpeg" || value == "full") {
        return "full";
      }
      throw worker_error("unsupported HDR color range: " + range);
    }

    std::string hdr_decode_filter(const media_contract_t &media) {
      if (media.color == media_color_e::sdr) {
        throw worker_error("HDR decode filter requested for SDR");
      }
      return "zscale=rangein=" + zscale_range(media.color_range) +
             ":primariesin=" + media.color_primaries +
             ":transferin=" + media.color_transfer +
             ":matrixin=" + media.color_space +
             ":range=full:primaries=bt709:transfer=linear:matrix=gbr:npl=80,"
             "format=gbrpf32le";
    }

    std::string hdr_encode_filter(const media_contract_t &media) {
      if (media.color == media_color_e::sdr) {
        throw worker_error("HDR encode filter requested for SDR");
      }
      return "format=gbrpf32le,"
             "zscale=rangein=full:primariesin=bt709:transferin=linear:matrixin=gbr:"
             "range=" +
             zscale_range(media.color_range) +
             ":primaries=" + media.color_primaries +
             ":transfer=" + media.color_transfer +
             ":matrix=" + media.color_space +
             ":npl=80,format=p010le";
    }

    std::wstring quote_windows_argument(const std::wstring_view argument) {
      std::wstring result;
      result.push_back(L'"');
      std::size_t backslashes = 0;
      for (const wchar_t character : argument) {
        if (character == L'\\') {
          ++backslashes;
          continue;
        }
        if (character == L'"') {
          result.append(backslashes * 2 + 1, L'\\');
          result.push_back(L'"');
          backslashes = 0;
          continue;
        }
        result.append(backslashes, L'\\');
        backslashes = 0;
        result.push_back(character);
      }
      result.append(backslashes * 2, L'\\');
      result.push_back(L'"');
      return result;
    }

    constexpr std::size_t max_child_log_bytes =
      8ull * 1024ull * 1024ull;
    constexpr std::size_t child_log_marker_reserve = 128;
    static_assert(max_child_log_bytes > child_log_marker_reserve + 1024);

    class bounded_log_accumulator_t {
    public:
      explicit bounded_log_accumulator_t(
        const std::size_t max_output_bytes = max_child_log_bytes
      ):
          max_output_bytes_(max_output_bytes),
          payload_capacity_(
            max_output_bytes > child_log_marker_reserve ?
              max_output_bytes - child_log_marker_reserve :
              0
          ),
          prefix_capacity_(
            std::min<std::size_t>(64ull * 1024ull, payload_capacity_ / 8)
          ),
          tail_capacity_(payload_capacity_ - prefix_capacity_),
          tail_(tail_capacity_) {
        if (tail_capacity_ == 0) {
          throw worker_error("bounded child log capacity is invalid");
        }
        prefix_.reserve(prefix_capacity_);
      }

      void append(const char *bytes, std::size_t size) {
        if (!bytes && size != 0) {
          throw worker_error("bounded child log received a null payload");
        }
        if (size == 0) {
          return;
        }
        total_bytes_ += size;
        const auto prefix_bytes =
          std::min<std::size_t>(size, prefix_capacity_ - prefix_.size());
        prefix_.insert(
          prefix_.end(),
          bytes,
          bytes + prefix_bytes
        );
        bytes += prefix_bytes;
        size -= prefix_bytes;

        while (size != 0) {
          const auto chunk = std::min<std::size_t>(
            size,
            tail_capacity_ - tail_write_
          );
          std::memcpy(tail_.data() + tail_write_, bytes, chunk);
          tail_write_ = (tail_write_ + chunk) % tail_capacity_;
          tail_size_ = std::min<std::size_t>(
            tail_capacity_,
            tail_size_ + chunk
          );
          bytes += chunk;
          size -= chunk;
        }
      }

      [[nodiscard]] std::vector<std::uint8_t> render() const {
        const auto retained_payload = prefix_.size() + tail_size_;
        const auto dropped =
          total_bytes_ > retained_payload ?
            total_bytes_ - retained_payload :
            0;
        const auto marker =
          dropped == 0 ?
            std::string {} :
            "\n[Sunshine 3D: " + std::to_string(dropped) +
              " child-log bytes omitted; showing prefix and tail]\n";
        if (marker.size() > child_log_marker_reserve) {
          throw worker_error("bounded child log marker reserve is too small");
        }

        std::vector<std::uint8_t> output;
        output.reserve(prefix_.size() + marker.size() + tail_size_);
        output.insert(output.end(), prefix_.begin(), prefix_.end());
        output.insert(output.end(), marker.begin(), marker.end());
        if (tail_size_ != 0) {
          const auto oldest =
            tail_size_ == tail_capacity_ ? tail_write_ : 0;
          const auto first = std::min<std::size_t>(
            tail_size_,
            tail_capacity_ - oldest
          );
          output.insert(
            output.end(),
            tail_.begin() + static_cast<std::ptrdiff_t>(oldest),
            tail_.begin() + static_cast<std::ptrdiff_t>(oldest + first)
          );
          output.insert(
            output.end(),
            tail_.begin(),
            tail_.begin() +
              static_cast<std::ptrdiff_t>(tail_size_ - first)
          );
        }
        if (output.size() > max_output_bytes_) {
          throw worker_error("bounded child log exceeded its output contract");
        }
        return output;
      }

    private:
      std::size_t max_output_bytes_ = 0;
      std::size_t payload_capacity_ = 0;
      std::size_t prefix_capacity_ = 0;
      std::size_t tail_capacity_ = 0;
      std::vector<std::uint8_t> prefix_;
      std::vector<std::uint8_t> tail_;
      std::size_t tail_write_ = 0;
      std::size_t tail_size_ = 0;
      std::uint64_t total_bytes_ = 0;
    };

#ifdef _WIN32
    [[nodiscard]] bool native_stdout_pipe_error_is_eof(
      const DWORD error
    ) noexcept {
      // A child can close its inherited stdout handle just before its process handle
      // becomes signaled.  Pipe closure is EOF regardless of that short-lived process
      // state; the caller still waits for and validates the child's exit status.
      return error == ERROR_BROKEN_PIPE || error == ERROR_NO_DATA;
    }

    class bounded_child_log_t {
    public:
      explicit bounded_child_log_t(fs::path path):
          path_(std::move(path)) {
        SECURITY_ATTRIBUTES inheritable {
          .nLength = sizeof(SECURITY_ATTRIBUTES),
          .lpSecurityDescriptor = nullptr,
          .bInheritHandle = TRUE,
        };
        if (
          !CreatePipe(
            &read_,
            &write_,
            &inheritable,
            1024 * 1024
          ) ||
          !SetHandleInformation(read_, HANDLE_FLAG_INHERIT, 0)
        ) {
          const auto error = GetLastError();
          close_handle(read_);
          close_handle(write_);
          throw worker_error(
            "cannot create bounded child-log pipe (Windows error " +
            std::to_string(error) + ")"
          );
        }
        try {
          drain_ = std::thread([this]() {
            drain_pipe();
          });
        } catch (...) {
          close_handle(read_);
          close_handle(write_);
          throw;
        }
      }

      bounded_child_log_t(const bounded_child_log_t &) = delete;
      bounded_child_log_t &operator=(const bounded_child_log_t &) = delete;

      ~bounded_child_log_t() {
        finish();
      }

      [[nodiscard]] HANDLE write_handle() const {
        return write_;
      }

      void close_parent_writer() noexcept {
        close_handle(write_);
      }

      void finish() noexcept {
        if (finished_) {
          return;
        }
        close_parent_writer();
        if (drain_.joinable()) {
          drain_.join();
        }
        close_handle(read_);
        try {
          if (read_error_ != ERROR_SUCCESS) {
            const auto note =
              "\n[Sunshine 3D: child-log pipe read failed with Windows error " +
              std::to_string(read_error_) + "]\n";
            accumulator_.append(note.data(), note.size());
          }
          write_bytes_atomic(path_, accumulator_.render());
        } catch (...) {
          // Process teardown is noexcept. A missing log must never strand a child/job; all
          // actual media/result contracts are published independently and remain authoritative.
        }
        finished_ = true;
      }

    private:
      static void close_handle(HANDLE &handle) noexcept {
        if (handle && handle != INVALID_HANDLE_VALUE) {
          CloseHandle(handle);
        }
        handle = nullptr;
      }

      void drain_pipe() noexcept {
        std::array<char, 64ull * 1024ull> buffer {};
        while (true) {
          DWORD received = 0;
          if (
            ReadFile(
              read_,
              buffer.data(),
              static_cast<DWORD>(buffer.size()),
              &received,
              nullptr
            )
          ) {
            if (received == 0) {
              return;
            }
            try {
              accumulator_.append(buffer.data(), received);
            } catch (...) {
              read_error_ = ERROR_NOT_ENOUGH_MEMORY;
              // Keep draining after an allocation failure so a verbose child cannot block
              // forever on a full pipe while its supervisor is trying to terminate it.
            }
            continue;
          }
          const auto error = GetLastError();
          if (error != ERROR_BROKEN_PIPE && error != ERROR_NO_DATA) {
            read_error_ = error;
          }
          return;
        }
      }

      fs::path path_;
      HANDLE read_ = nullptr;
      HANDLE write_ = nullptr;
      std::thread drain_;
      bounded_log_accumulator_t accumulator_;
      DWORD read_error_ = ERROR_SUCCESS;
      bool finished_ = false;
    };

    std::wstring command_line(const std::vector<std::string> &arguments) {
      std::wstring result;
      for (const auto &argument : arguments) {
        if (!result.empty()) {
          result.push_back(L' ');
        }
        result += quote_windows_argument(path_from_utf8(argument).wstring());
      }
      return result;
    }

    void validate_windows_command_line_capacity(
      const std::vector<std::string> &arguments,
      const std::string_view description
    ) {
      // CreateProcessW's 32,767-character limit includes the terminating NUL.
      constexpr std::size_t max_command_characters = 32766;
      if (command_line(arguments).size() > max_command_characters) {
        throw worker_error(
          std::string(description) +
          " exceeds the Windows command-line limit"
        );
      }
    }

    class child_process_t {
    public:
      child_process_t() = default;
      child_process_t(const child_process_t &) = delete;
      child_process_t &operator=(const child_process_t &) = delete;

      child_process_t(child_process_t &&other) noexcept {
        *this = std::move(other);
      }

      child_process_t &operator=(child_process_t &&other) noexcept {
        if (this != &other) {
          terminate();
          process_ = std::exchange(other.process_, nullptr);
          thread_ = std::exchange(other.thread_, nullptr);
          stdout_read_ = std::exchange(other.stdout_read_, nullptr);
          stdout_file_ = std::exchange(other.stdout_file_, nullptr);
          log_capture_ = std::move(other.log_capture_);
          stdin_null_ = std::exchange(other.stdin_null_, nullptr);
          job_ = std::exchange(other.job_, nullptr);
        }
        return *this;
      }

      ~child_process_t() {
        terminate();
      }

      static child_process_t launch(
        const std::vector<std::string> &arguments,
        const fs::path &working_directory,
        const fs::path &log_path,
        const std::optional<fs::path> &stdout_path = std::nullopt,
        const bool pipe_stdout = false
      ) {
        if (arguments.empty()) {
          throw worker_error("cannot launch an empty command");
        }
        validate_windows_command_line_capacity(
          arguments,
          "native child command"
        );
        std::error_code ec;
        fs::create_directories(log_path.parent_path(), ec);
        if (ec) {
          throw worker_error("cannot create process-log directory");
        }

        child_process_t child;
        SECURITY_ATTRIBUTES inheritable {
          .nLength = sizeof(SECURITY_ATTRIBUTES),
          .lpSecurityDescriptor = nullptr,
          .bInheritHandle = TRUE,
        };
        child.log_capture_ =
          std::make_unique<bounded_child_log_t>(log_path);
        const HANDLE log_write = child.log_capture_->write_handle();

        HANDLE stdout_write = log_write;
        if (pipe_stdout) {
          if (!CreatePipe(&child.stdout_read_, &stdout_write, &inheritable, 1024 * 1024) || !SetHandleInformation(child.stdout_read_, HANDLE_FLAG_INHERIT, 0)) {
            if (stdout_write && stdout_write != log_write) {
              CloseHandle(stdout_write);
            }
            throw worker_error("cannot create bounded decoder pipe");
          }
        } else if (stdout_path) {
          fs::create_directories(stdout_path->parent_path(), ec);
          child.stdout_file_ = CreateFileW(
            stdout_path->c_str(),
            GENERIC_WRITE,
            FILE_SHARE_READ,
            &inheritable,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr
          );
          if (child.stdout_file_ == INVALID_HANDLE_VALUE) {
            child.stdout_file_ = nullptr;
            throw worker_error("cannot create child stdout file");
          }
          stdout_write = child.stdout_file_;
        }

        child.stdin_null_ = CreateFileW(
          L"NUL",
          GENERIC_READ,
          FILE_SHARE_READ | FILE_SHARE_WRITE,
          &inheritable,
          OPEN_EXISTING,
          FILE_ATTRIBUTE_NORMAL,
          nullptr
        );
        if (child.stdin_null_ == INVALID_HANDLE_VALUE) {
          child.stdin_null_ = nullptr;
          throw worker_error("cannot open the null device for child stdin");
        }
        std::vector<HANDLE> inherited_handles {
          child.stdin_null_,
          stdout_write,
          log_write,
        };
        std::sort(inherited_handles.begin(), inherited_handles.end());
        inherited_handles.erase(
          std::unique(inherited_handles.begin(), inherited_handles.end()),
          inherited_handles.end()
        );
        SIZE_T attribute_bytes = 0;
        InitializeProcThreadAttributeList(
          nullptr,
          1,
          0,
          &attribute_bytes
        );
        std::vector<std::uint8_t> attribute_storage(attribute_bytes);
        auto *attribute_list =
          reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(
            attribute_storage.data()
          );
        bool attribute_initialized = false;
        if (attribute_bytes != 0) {
          attribute_initialized = InitializeProcThreadAttributeList(
            attribute_list,
            1,
            0,
            &attribute_bytes
          );
        }
        if (!attribute_initialized || !UpdateProcThreadAttribute(attribute_list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited_handles.data(), inherited_handles.size() * sizeof(HANDLE), nullptr, nullptr)) {
          if (attribute_initialized) {
            DeleteProcThreadAttributeList(attribute_list);
          }
          throw worker_error(
            "cannot restrict native child handle inheritance"
          );
        }
        STARTUPINFOEXW startup {};
        startup.StartupInfo.cb = sizeof(startup);
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput = child.stdin_null_;
        startup.StartupInfo.hStdOutput = stdout_write;
        startup.StartupInfo.hStdError = log_write;
        startup.lpAttributeList = attribute_list;
        PROCESS_INFORMATION process {};
        auto line = command_line(arguments);
        std::vector<wchar_t> mutable_line(line.begin(), line.end());
        mutable_line.push_back(L'\0');
        const auto application = path_from_utf8(arguments.front()).wstring();
        const auto cwd = working_directory.wstring();
        const BOOL launched = CreateProcessW(
          application.c_str(),
          mutable_line.data(),
          nullptr,
          nullptr,
          TRUE,
          CREATE_NO_WINDOW | CREATE_SUSPENDED |
            EXTENDED_STARTUPINFO_PRESENT,
          nullptr,
          cwd.empty() ? nullptr : cwd.c_str(),
          &startup.StartupInfo,
          &process
        );
        const auto launch_error =
          launched ? ERROR_SUCCESS : GetLastError();
        DeleteProcThreadAttributeList(attribute_list);
        if (pipe_stdout && stdout_write && stdout_write != log_write) {
          CloseHandle(stdout_write);
        }
        // CreateProcess has either inherited its own write handle or failed. Closing the
        // parent's copy lets the drain thread observe EOF as soon as the process tree exits.
        child.log_capture_->close_parent_writer();
        if (!launched) {
          throw worker_error(
            "cannot launch native child process (Windows error " +
            std::to_string(launch_error) + ")"
          );
        }
        child.job_ = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION job_limits {};
        job_limits.BasicLimitInformation.LimitFlags =
          JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!child.job_ || !SetInformationJobObject(child.job_, JobObjectExtendedLimitInformation, &job_limits, sizeof(job_limits)) || !AssignProcessToJobObject(child.job_, process.hProcess) || ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
          const auto error = GetLastError();
          TerminateProcess(process.hProcess, 9);
          CloseHandle(process.hProcess);
          CloseHandle(process.hThread);
          if (child.job_) {
            CloseHandle(child.job_);
            child.job_ = nullptr;
          }
          throw worker_error(
            "cannot establish kill-on-close child ownership (Windows error " +
            std::to_string(error) + ")"
          );
        }
        child.process_ = process.hProcess;
        child.thread_ = process.hThread;
        return child;
      }

      [[nodiscard]] bool running() const {
        return process_ &&
               WaitForSingleObject(process_, 0) == WAIT_TIMEOUT;
      }

      [[nodiscard]] int exit_code() const {
        if (!process_) {
          return -1;
        }
        DWORD code = 0;
        if (!GetExitCodeProcess(process_, &code)) {
          return -1;
        }
        return static_cast<int>(code);
      }

      void read_exact(void *destination, const std::size_t size) {
        if (!stdout_read_) {
          throw worker_error("child process has no stdout pipe");
        }
        auto *bytes = static_cast<std::uint8_t *>(destination);
        std::size_t offset = 0;
        const auto deadline =
          std::chrono::steady_clock::now() + frame_io_timeout;
        while (offset < size) {
          DWORD available = 0;
          if (!PeekNamedPipe(
                stdout_read_,
                nullptr,
                0,
                nullptr,
                &available,
                nullptr
              )) {
            throw worker_error(
              "streaming decoder pipe failed inside a frame (" +
              std::to_string(offset) + " of " + std::to_string(size) +
              " bytes)"
            );
          }
          if (available == 0) {
            if (!running()) {
              throw worker_error(
                "streaming decoder ended inside a frame (" +
                std::to_string(offset) + " of " + std::to_string(size) +
                " bytes)"
              );
            }
            if (std::chrono::steady_clock::now() >= deadline) {
              throw worker_error(
                "streaming decoder timed out inside a frame (" +
                std::to_string(offset) + " of " + std::to_string(size) +
                " bytes)"
              );
            }
            std::this_thread::sleep_for(child_poll);
            continue;
          }
          const DWORD request = static_cast<DWORD>(
            std::min<std::size_t>(
              std::min<std::size_t>(
                size - offset,
                std::numeric_limits<DWORD>::max()
              ),
              available
            )
          );
          DWORD received = 0;
          if (!ReadFile(stdout_read_, bytes + offset, request, &received, nullptr) || received == 0) {
            throw worker_error(
              "streaming decoder ended inside a frame (" +
              std::to_string(offset) + " of " + std::to_string(size) +
              " bytes)"
            );
          }
          offset += received;
        }
      }

      std::string read_line(const std::size_t limit) {
        std::string line;
        while (line.size() < limit) {
          char character = 0;
          read_exact(&character, 1);
          line.push_back(character);
          if (character == '\n') {
            return line;
          }
        }
        throw worker_error("decoder emitted an overlong frame header");
      }

      std::size_t read_some(
        void *destination,
        const std::size_t capacity,
        const std::chrono::steady_clock::time_point absolute_deadline
      ) {
        if (!stdout_read_ || capacity == 0) {
          throw worker_error("child process has no readable stdout pipe");
        }
        const auto idle_deadline = std::min(
          std::chrono::steady_clock::now() + frame_io_timeout,
          absolute_deadline
        );
        while (true) {
          DWORD available = 0;
          if (!PeekNamedPipe(
                stdout_read_,
                nullptr,
                0,
                nullptr,
                &available,
                nullptr
              )) {
            const auto error = GetLastError();
            if (native_stdout_pipe_error_is_eof(error)) {
              return 0;
            }
            throw worker_error("native child stdout pipe failed");
          }
          if (available != 0) {
            const auto request = static_cast<DWORD>(
              std::min<std::size_t>(
                std::min<std::size_t>(
                  capacity,
                  std::numeric_limits<DWORD>::max()
                ),
                available
              )
            );
            DWORD received = 0;
            if (
              !ReadFile(
                stdout_read_,
                destination,
                request,
                &received,
                nullptr
              )
            ) {
              const auto error = GetLastError();
              if (native_stdout_pipe_error_is_eof(error)) {
                return 0;
              }
              throw worker_error("native child stdout pipe read failed");
            }
            if (received == 0) {
              throw worker_error("native child stdout pipe made no progress");
            }
            return received;
          }
          if (!running()) {
            return 0;
          }
          if (std::chrono::steady_clock::now() >= idle_deadline) {
            throw worker_error("native child stdout pipe timed out");
          }
          std::this_thread::sleep_for(child_poll);
        }
      }

      bool read_one_if_available() {
        if (!stdout_read_) {
          return false;
        }
        DWORD available = 0;
        if (!PeekNamedPipe(
              stdout_read_,
              nullptr,
              0,
              nullptr,
              &available,
              nullptr
            )) {
          return false;
        }
        if (available == 0) {
          return false;
        }
        char byte = 0;
        read_exact(&byte, 1);
        return true;
      }

      int wait(const std::chrono::milliseconds timeout) {
        if (!process_) {
          throw worker_error("cannot wait for an invalid child");
        }
        const DWORD wait_ms = timeout.count() >
                                  std::numeric_limits<DWORD>::max() ?
                                INFINITE :
                                static_cast<DWORD>(timeout.count());
        if (WaitForSingleObject(process_, wait_ms) != WAIT_OBJECT_0) {
          throw worker_error("native child process timed out");
        }
        return exit_code();
      }

      void terminate() noexcept {
        if (job_) {
          TerminateJobObject(job_, 9);
        }
        if (process_) {
          if (WaitForSingleObject(process_, 0) == WAIT_TIMEOUT) {
            TerminateProcess(process_, 9);
            WaitForSingleObject(process_, 5000);
          }
          CloseHandle(process_);
          process_ = nullptr;
        }
        if (thread_) {
          CloseHandle(thread_);
          thread_ = nullptr;
        }
        if (stdout_read_) {
          CloseHandle(stdout_read_);
          stdout_read_ = nullptr;
        }
        if (stdout_file_) {
          CloseHandle(stdout_file_);
          stdout_file_ = nullptr;
        }
        if (stdin_null_) {
          CloseHandle(stdin_null_);
          stdin_null_ = nullptr;
        }
        if (job_) {
          CloseHandle(job_);
          job_ = nullptr;
        }
        if (log_capture_) {
          log_capture_->finish();
          log_capture_.reset();
        }
      }

    private:
      HANDLE process_ = nullptr;
      HANDLE thread_ = nullptr;
      HANDLE stdout_read_ = nullptr;
      HANDLE stdout_file_ = nullptr;
      std::unique_ptr<bounded_child_log_t> log_capture_;
      HANDLE stdin_null_ = nullptr;
      HANDLE job_ = nullptr;
    };
#else
    class child_process_t {
    public:
      static child_process_t launch(
        const std::vector<std::string> &,
        const fs::path &,
        const fs::path &,
        const std::optional<fs::path> & = std::nullopt,
        bool = false
      ) {
        throw worker_error("offline SBS conversion is Windows-only");
      }

      bool running() const {
        return false;
      }

      int exit_code() const {
        return -1;
      }

      void read_exact(void *, std::size_t) {}

      std::string read_line(std::size_t) {
        return {};
      }

      std::size_t read_some(
        void *,
        std::size_t,
        std::chrono::steady_clock::time_point
      ) {
        return 0;
      }

      bool read_one_if_available() {
        return false;
      }

      int wait(std::chrono::milliseconds) {
        return -1;
      }

      void terminate() {}
    };
#endif

#ifdef _WIN32
    class scene_frame_server_t {
    public:
      scene_frame_server_t(
        const std::uint64_t first_sequence,
        const std::uint64_t end_sequence_exclusive,
        std::string extension
      ):
          first_sequence_(first_sequence),
          end_sequence_exclusive_(end_sequence_exclusive),
          extension_(std::move(extension)) {
        if (first_sequence_ == 0 || end_sequence_exclusive_ <= first_sequence_ || (extension_ != "png" && extension_ != "pfm")) {
          throw worker_error("invalid scene HTTP bridge contract");
        }
        WSADATA data {};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
          throw worker_error("cannot initialize the loopback frame bridge");
        }
        winsock_started_ = true;
        try {
          listener_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
          if (listener_ == INVALID_SOCKET) {
            throw worker_error("cannot create the loopback frame bridge socket");
          }
          BOOL exclusive = TRUE;
          if (setsockopt(listener_, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, reinterpret_cast<const char *>(&exclusive), sizeof(exclusive)) != 0) {
            throw worker_error("cannot make the loopback frame bridge exclusive");
          }
          sockaddr_in address {};
          address.sin_family = AF_INET;
          address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
          address.sin_port = 0;
          if (bind(listener_, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0 || listen(listener_, 1) != 0) {
            throw worker_error("cannot bind the loopback frame bridge");
          }
          int address_size = sizeof(address);
          if (getsockname(listener_, reinterpret_cast<sockaddr *>(&address), &address_size) != 0 || ntohs(address.sin_port) == 0) {
            throw worker_error("cannot resolve the loopback frame bridge port");
          }
          port_ = ntohs(address.sin_port);
          token_ = secure_frame_bridge_token();
          next_request_ = first_sequence_;
          next_publish_ = first_sequence_;
          thread_ = std::thread([this] {
            serve();
          });
        } catch (...) {
          if (listener_ != INVALID_SOCKET) {
            closesocket(listener_);
            listener_ = INVALID_SOCKET;
          }
          cleanup_winsock();
          throw;
        }
      }

      scene_frame_server_t(const scene_frame_server_t &) = delete;
      scene_frame_server_t &operator=(const scene_frame_server_t &) = delete;

      ~scene_frame_server_t() {
        abort("scene bridge destroyed");
      }

      [[nodiscard]] std::string frame_url(
        const std::uint64_t sequence
      ) const {
        if (sequence < first_sequence_ || sequence >= end_sequence_exclusive_) {
          throw worker_error("scene bridge URL sequence is out of range");
        }
        return "http://127.0.0.1:" + std::to_string(port_) + "/" + token_ +
               "/frame_" + frame_id(sequence) + "." + extension_;
      }

#ifdef SUNSHINE_TESTS
      [[nodiscard]] std::uint16_t port_for_test() const {
        return port_;
      }
#endif

      void publish_and_wait(
        const std::uint64_t sequence,
        const fs::path &path,
        const child_process_t &encoder
      ) {
        std::error_code ec;
        if (sequence < first_sequence_ || sequence >= end_sequence_exclusive_ || !fs::is_regular_file(path, ec) || ec || fs::file_size(path, ec) == 0 || ec) {
          throw worker_error("cannot publish an invalid SBS bridge frame");
        }
        std::unique_lock lock(mutex_);
        if (!error_.empty()) {
          throw worker_error("scene bridge failed: " + error_);
        }
        if (sequence != next_publish_ || published_path_) {
          throw worker_error("scene bridge publication is not monotonic");
        }
        published_sequence_ = sequence;
        published_path_ = path;
        ++next_publish_;
        condition_.notify_all();
        const auto deadline =
          std::chrono::steady_clock::now() + frame_io_timeout;
        while (served_sequence_ < sequence && error_.empty()) {
          if (!encoder.running()) {
            error_ =
              "whole-clip encoder exited before requesting frame " +
              std::to_string(sequence) + " (exit " +
              std::to_string(encoder.exit_code()) + ")";
            break;
          }
          if (std::chrono::steady_clock::now() >= deadline) {
            error_ = "timed out serving scene frame " +
                     std::to_string(sequence);
            break;
          }
          condition_.wait_for(lock, child_poll);
        }
        if (!error_.empty()) {
          throw worker_error("scene bridge failed: " + error_);
        }
      }

      void finish() {
        {
          std::unique_lock lock(mutex_);
          const auto expected = end_sequence_exclusive_ - 1;
          if (!error_.empty()) {
            throw worker_error("scene bridge failed: " + error_);
          }
          if (served_sequence_ != expected || next_publish_ != end_sequence_exclusive_) {
            throw worker_error(
              "whole-clip encoder did not consume the complete SBS sequence"
            );
          }
          stopping_ = true;
          condition_.notify_all();
        }
        close_sockets();
        if (thread_.joinable()) {
          thread_.join();
        }
        cleanup_winsock();
      }

      void abort(const std::string &reason) noexcept {
        {
          std::lock_guard lock(mutex_);
          if (error_.empty()) {
            error_ = reason;
          }
          stopping_ = true;
          condition_.notify_all();
        }
        close_sockets();
        if (thread_.joinable()) {
          thread_.join();
        }
        cleanup_winsock();
      }

    private:
      static bool send_all(
        const SOCKET socket_handle,
        const void *data,
        const std::size_t size
      ) {
        const auto *bytes = static_cast<const char *>(data);
        std::size_t offset = 0;
        while (offset < size) {
          const int chunk = static_cast<int>(
            std::min<std::size_t>(
              size - offset,
              static_cast<std::size_t>(
                std::numeric_limits<int>::max()
              )
            )
          );
          const int written = send(
            socket_handle,
            bytes + offset,
            chunk,
            0
          );
          if (written <= 0) {
            return false;
          }
          offset += static_cast<std::size_t>(written);
        }
        return true;
      }

      void close_sockets() noexcept {
        const auto close_one = [](SOCKET &socket_handle) {
          if (socket_handle != INVALID_SOCKET) {
            shutdown(socket_handle, SD_BOTH);
            closesocket(socket_handle);
            socket_handle = INVALID_SOCKET;
          }
        };
        std::lock_guard socket_lock(socket_mutex_);
        close_one(active_client_);
        close_one(listener_);
      }

      void cleanup_winsock() noexcept {
        if (winsock_started_) {
          WSACleanup();
          winsock_started_ = false;
        }
      }

      std::string receive_request(const SOCKET client) {
        // A capability-bearing local FFmpeg request is emitted immediately. Keep the
        // unauthenticated header window short so an unrelated loopback process cannot pin the
        // single scene bridge for the lifetime of a frame.
        DWORD timeout_ms = 2000;
        setsockopt(
          client,
          SOL_SOCKET,
          SO_RCVTIMEO,
          reinterpret_cast<const char *>(&timeout_ms),
          sizeof(timeout_ms)
        );
        std::string request;
        std::array<char, 2048> buffer {};
        while (request.find("\r\n\r\n") == std::string::npos) {
          if (request.size() >= 8192) {
            throw worker_error("scene bridge received an oversized request");
          }
          const int received = recv(
            client,
            buffer.data(),
            static_cast<int>(
              std::min<std::size_t>(
                buffer.size(),
                8192 - request.size()
              )
            ),
            0
          );
          if (received <= 0) {
            throw worker_error("scene bridge request ended early");
          }
          request.append(buffer.data(), static_cast<std::size_t>(received));
        }
        return request;
      }

      void send_error(
        const SOCKET client,
        const int status,
        const std::string_view reason
      ) {
        const auto response =
          "HTTP/1.1 " + std::to_string(status) + " " +
          std::string(reason) +
          "\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
        send_all(client, response.data(), response.size());
      }

      bool serve_frame(
        const SOCKET client,
        const fs::path &path,
        const bool body
      ) {
        std::error_code ec;
        const auto size = fs::file_size(path, ec);
        if (ec || size == 0) {
          return false;
        }
        const std::string header =
          "HTTP/1.1 200 OK\r\n"
          "Content-Type: " +
          std::string(
            extension_ == "png" ?
              "image/png" :
              "image/x-portable-floatmap"
          ) +
          "\r\nContent-Length: " + std::to_string(size) +
          "\r\nCache-Control: no-store\r\n"
          "X-Content-Type-Options: nosniff\r\n"
          "Connection: close\r\n\r\n";
        if (!send_all(client, header.data(), header.size())) {
          return false;
        }
        if (!body) {
          return true;
        }
        std::ifstream stream(path, std::ios::binary);
        std::vector<char> buffer(1024 * 1024);
        std::uintmax_t sent = 0;
        while (stream && sent < size) {
          const auto wanted = static_cast<std::streamsize>(
            std::min<std::uintmax_t>(buffer.size(), size - sent)
          );
          stream.read(buffer.data(), wanted);
          const auto count = stream.gcount();
          if (count <= 0 || !send_all(client, buffer.data(), static_cast<std::size_t>(count))) {
            return false;
          }
          sent += static_cast<std::uintmax_t>(count);
        }
        return sent == size;
      }

      void fail(const std::string &message) noexcept {
        std::lock_guard lock(mutex_);
        if (error_.empty()) {
          error_ = message;
        }
        stopping_ = true;
        condition_.notify_all();
      }

      void serve() noexcept {
        try {
          while (next_request_ < end_sequence_exclusive_) {
            SOCKET listener = INVALID_SOCKET;
            {
              std::lock_guard socket_lock(socket_mutex_);
              listener = listener_;
            }
            SOCKET client = accept(listener, nullptr, nullptr);
            if (client == INVALID_SOCKET) {
              std::lock_guard lock(mutex_);
              if (stopping_) {
                return;
              }
              throw worker_error("scene bridge accept failed");
            }
            {
              std::lock_guard socket_lock(socket_mutex_);
              active_client_ = client;
            }
            const auto close_client = [&] {
              std::lock_guard socket_lock(socket_mutex_);
              if (active_client_ != INVALID_SOCKET) {
                shutdown(active_client_, SD_BOTH);
                closesocket(active_client_);
                active_client_ = INVALID_SOCKET;
              }
            };
            std::string request;
            try {
              request = receive_request(client);
            } catch (const std::exception &) {
              // No complete request line means no capability was authenticated. A local port
              // scanner, early close, oversized header, or idle socket is isolated to this
              // connection and must never fail the conversion job.
              send_error(client, 400, "Bad Request");
              close_client();
              continue;
            }
            const auto line_end = request.find("\r\n");
            const std::string expected_path =
              "/" + token_ + "/frame_" + frame_id(next_request_) +
              "." + extension_;
            const std::string expected_get =
              "GET " + expected_path + " HTTP/1.1";
            const std::string expected_head =
              "HEAD " + expected_path + " HTTP/1.1";
            if (line_end == std::string::npos || (request.substr(0, line_end) != expected_get && request.substr(0, line_end) != expected_head)) {
              send_error(client, 404, "Not Found");
              close_client();
              continue;
            }
            const bool head_request =
              request.substr(0, line_end) == expected_head;
            fs::path published;
            {
              std::unique_lock lock(mutex_);
              condition_.wait(lock, [&] {
                return stopping_ ||
                       (!error_.empty()) ||
                       (published_path_ &&
                        published_sequence_ == next_request_);
              });
              if (stopping_ || !error_.empty()) {
                send_error(client, 503, "Unavailable");
                close_client();
                return;
              }
              published = *published_path_;
            }
            if (!serve_frame(client, published, !head_request)) {
              close_client();
              throw worker_error(
                "scene bridge failed while serving frame " +
                std::to_string(next_request_)
              );
            }
            close_client();
            if (head_request) {
              continue;
            }
            {
              std::lock_guard lock(mutex_);
              served_sequence_ = next_request_;
              published_path_.reset();
              ++next_request_;
              condition_.notify_all();
            }
          }
        } catch (const std::exception &exception) {
          fail(exception.what());
        }
      }

      std::uint64_t first_sequence_ = 0;
      std::uint64_t end_sequence_exclusive_ = 0;
      std::string extension_;
      std::uint16_t port_ = 0;
      std::string token_;
      SOCKET listener_ = INVALID_SOCKET;
      SOCKET active_client_ = INVALID_SOCKET;
      bool winsock_started_ = false;
      std::thread thread_;
      std::mutex mutex_;
      std::condition_variable condition_;
      std::mutex socket_mutex_;
      bool stopping_ = false;
      std::string error_;
      std::uint64_t next_request_ = 0;
      std::uint64_t next_publish_ = 0;
      std::uint64_t published_sequence_ = 0;
      std::uint64_t served_sequence_ = 0;
      std::optional<fs::path> published_path_;
    };
#else
    class scene_frame_server_t {
    public:
      scene_frame_server_t(std::uint64_t, std::uint64_t, std::string) {
        throw worker_error("offline SBS conversion is Windows-only");
      }

      std::string frame_url(std::uint64_t) const {
        throw worker_error("offline SBS conversion is Windows-only");
      }

      void publish_and_wait(
        std::uint64_t,
        const fs::path &,
        const child_process_t &
      ) {
        throw worker_error("offline SBS conversion is Windows-only");
      }

      void finish() {
        throw worker_error("offline SBS conversion is Windows-only");
      }

      void abort(const std::string &) noexcept {}
    };
#endif

    class child_stdout_streambuf_t: public std::streambuf {
    public:
      child_stdout_streambuf_t(
        child_process_t &child,
        const std::chrono::steady_clock::time_point absolute_deadline
      ):
          child_(child),
          absolute_deadline_(absolute_deadline) {
        setg(buffer_.data(), buffer_.data(), buffer_.data());
      }

    protected:
      int_type underflow() override {
        if (gptr() < egptr()) {
          return traits_type::to_int_type(*gptr());
        }
        if (std::chrono::steady_clock::now() >= absolute_deadline_) {
          throw worker_error("native child process timed out");
        }
        const auto received = child_.read_some(
          buffer_.data(),
          buffer_.size(),
          absolute_deadline_
        );
        if (received == 0) {
          return traits_type::eof();
        }
        setg(
          buffer_.data(),
          buffer_.data(),
          buffer_.data() + received
        );
        return traits_type::to_int_type(*gptr());
      }

    private:
      child_process_t &child_;
      std::chrono::steady_clock::time_point absolute_deadline_;
      std::array<char, 64ull * 1024ull> buffer_ {};
    };

    void run_logged(
      const std::vector<std::string> &command,
      const fs::path &working_directory,
      const fs::path &log_path,
      const std::optional<fs::path> &stdout_path = std::nullopt
    ) {
      auto child = child_process_t::launch(
        command,
        working_directory,
        log_path,
        stdout_path
      );
      const int code = child.wait(
        std::chrono::duration_cast<std::chrono::milliseconds>(child_timeout)
      );
      if (code != 0) {
        throw worker_error(
          "child process exited " + std::to_string(code) +
          "; see " + path_utf8(log_path)
        );
      }
      child.terminate();
    }

    nlohmann::json run_logged_json_bounded(
      const std::vector<std::string> &command,
      const fs::path &working_directory,
      const fs::path &log_path,
      const std::uintmax_t max_bytes,
      const std::string_view description
    ) {
      if (max_bytes == 0 || max_bytes > std::string {}.max_size()) {
        throw worker_error(
          std::string(description) + " has an invalid byte bound"
        );
      }
      const auto deadline =
        std::chrono::steady_clock::now() + child_timeout;
      auto child = child_process_t::launch(
        command,
        working_directory,
        log_path,
        std::nullopt,
        true
      );
      std::string bytes;
      bytes.reserve(
        static_cast<std::size_t>(
          std::min<std::uintmax_t>(
            max_bytes,
            1024ull * 1024ull
          )
        )
      );
      std::array<char, 64ull * 1024ull> buffer {};
      while (true) {
        const auto received = child.read_some(
          buffer.data(),
          buffer.size(),
          deadline
        );
        if (received == 0) {
          break;
        }
        if (received > max_bytes - bytes.size()) {
          throw worker_error(
            std::string(description) + " exceeds its " +
            std::to_string(max_bytes) + "-byte contract"
          );
        }
        bytes.append(buffer.data(), received);
      }
      const auto now = std::chrono::steady_clock::now();
      const auto remaining = now < deadline ?
                               std::chrono::duration_cast<
                                 std::chrono::milliseconds
                               >(deadline - now) :
                               std::chrono::milliseconds {0};
      const int code = child.wait(remaining);
      if (code != 0) {
        throw worker_error(
          "child process exited " + std::to_string(code) +
          "; see " + path_utf8(log_path)
        );
      }
      child.terminate();
      if (bytes.empty()) {
        throw worker_error(std::string(description) + " is empty");
      }
      try {
        return parse_json_without_duplicate_keys(bytes);
      } catch (const std::exception &exception) {
        throw worker_error(
          "cannot parse " + std::string(description) + ": " +
          exception.what()
        );
      }
    }

    std::string timing_sha256(const media_contract_t &media) {
      const auto context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> {
        EVP_MD_CTX_new(),
        &EVP_MD_CTX_free,
      };
      if (
        !context ||
        EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1
      ) {
        throw worker_error("cannot initialize media timing SHA-256");
      }
      const auto update_u64 = [&](const std::uint64_t value) {
        std::array<std::uint8_t, sizeof(value)> bytes {};
        for (std::size_t index = 0; index < bytes.size(); ++index) {
          bytes[index] = static_cast<std::uint8_t>(
            (value >> (index * 8u)) & 0xffu
          );
        }
        if (
          EVP_DigestUpdate(
            context.get(),
            bytes.data(),
            bytes.size()
          ) != 1
        ) {
          throw worker_error("cannot update media timing SHA-256");
        }
      };
      update_u64(std::bit_cast<std::uint64_t>(media.time_base.numerator));
      update_u64(std::bit_cast<std::uint64_t>(media.time_base.denominator));
      update_u64(media.frames.size());
      for (const auto &frame : media.frames) {
        update_u64(frame.sequence);
        update_u64(std::bit_cast<std::uint64_t>(frame.pts));
        update_u64(std::bit_cast<std::uint64_t>(frame.duration));
      }
      std::array<unsigned char, EVP_MAX_MD_SIZE> digest {};
      unsigned int digest_size = 0;
      if (
        EVP_DigestFinal_ex(
          context.get(),
          digest.data(),
          &digest_size
        ) != 1 ||
        digest_size != 32
      ) {
        throw worker_error("cannot finalize media timing SHA-256");
      }
      static constexpr char digits[] = "0123456789abcdef";
      std::string result(digest_size * 2, '\0');
      for (std::size_t index = 0; index < digest_size; ++index) {
        result[index * 2] = digits[digest[index] >> 4u];
        result[index * 2 + 1] = digits[digest[index] & 0x0fu];
      }
      return result;
    }

    nlohmann::json media_contract_json(const media_contract_t &media) {
      if (media.frames.empty()) {
        throw worker_error("cannot serialize an empty media contract");
      }
      const auto [shortest, longest] = std::minmax_element(
        media.frames.begin(),
        media.frames.end(),
        [](const frame_timing_t &left, const frame_timing_t &right) {
          return left.duration < right.duration;
        }
      );
      const char *mode =
        media.color == media_color_e::sdr ? "sdr" :
                                            (media.color == media_color_e::hdr_pq ? "hdr-pq" : "hdr-hlg");
      return {
        {"schema", 2},
        {"width", media.width},
        {"height", media.height},
        {"codec_name", media.codec_name},
        {"pixel_format", media.pixel_format},
        {"color", {
                    {"mode", mode},
                    {"range", media.color_range},
                    {"space", media.color_space},
                    {"transfer", media.color_transfer},
                    {"primaries", media.color_primaries},
                    {"mastering_display", media.mastering_display},
                    {"content_light_level", media.content_light_level},
                    {"side_data_types", media.static_side_data_types},
                    {"dropped_nonsemantic_side_data_types", media.dropped_nonsemantic_side_data_types},
                  }},
        {"time_base", {
                        {"numerator", media.time_base.numerator},
                        {"denominator", media.time_base.denominator},
                      }},
        {"first_pts", media.first_pts()},
        {"end_pts_exclusive", media.end_pts_exclusive()},
        {"duration_seconds", media.duration_seconds()},
        {"variable_frame_rate", media.variable_frame_rate()},
        {"frame_count", media.frames.size()},
        {"timing", {
                     {"representation", "compact-summary-v1"},
                     {"sha256", timing_sha256(media)},
                     {"shortest_duration_ticks", shortest->duration},
                     {"longest_duration_ticks", longest->duration},
                     {"raw_ffprobe_retained", false},
                   }},
      };
    }

    void publish_progress(
      const worker_spec_t &spec,
      const std::string_view phase,
      const std::uint64_t processed_frames,
      const media_contract_t *media,
      const std::uint64_t scene_count,
      nlohmann::json current_scene = nullptr,
      nlohmann::json scene_decisions = nlohmann::json::array()
    ) {
      if (scene_count > max_serialized_scene_count) {
        throw worker_error(
          "worker progress scene count exceeds the serialized contract"
        );
      }
      if (!scene_decisions.is_array() || scene_decisions.size() > 32) {
        throw worker_error("worker progress scene decisions exceed their bound");
      }
      nlohmann::json value {
        {"schema", 1},
        {"job_id", spec.job_id},
        {"phase", phase},
        {"processed_frames", processed_frames},
        {"total_frames", media ? nlohmann::json(media->frames.size()) : nlohmann::json(nullptr)},
        {"source_time_seconds", media && processed_frames > 0 ? nlohmann::json(media->time_base.seconds(nonnegative_tick_difference(media->frames.at(std::min<std::size_t>(static_cast<std::size_t>(processed_frames), media->frames.size()) - 1).pts, media->first_pts(), "worker progress presentation offset"))) : nlohmann::json(nullptr)},
        {"source_duration_seconds", media ? nlohmann::json(media->duration_seconds()) : nlohmann::json(nullptr)},
        {"scene_count", scene_count},
        {"current_scene", std::move(current_scene)},
        {"scene_decisions", std::move(scene_decisions)},
      };
      write_json_atomic(spec.progress_path, value);
    }

    void publish_failure(
      const worker_spec_t &spec,
      const std::string &error
    ) noexcept {
      try {
        write_json_atomic(spec.result_path, {
                                              {"schema", 1},
                                              {"job_id", spec.job_id},
                                              {"status", "failed"},
                                              {"error", error.substr(0, 4096)},
                                              {"python_dependency", false},
                                            });
      } catch (...) {
      }
    }

    void write_text_atomic(const fs::path &path, const std::string &text) {
      write_bytes_atomic(
        path,
        std::vector<std::uint8_t>(text.begin(), text.end())
      );
    }

    nlohmann::json read_progress(
      const fs::path &path,
      const child_process_t &child,
      const std::uint64_t minimum,
      const std::string_view description
    ) {
      const auto deadline = std::chrono::steady_clock::now() + child_timeout;
      std::string last_parse_error;
      while (std::chrono::steady_clock::now() < deadline) {
        std::error_code ec;
        if (fs::is_regular_file(path, ec) && !ec) {
          try {
            auto value = read_json(path);
            if (value.value("schema", 0) != 1 || value.value("processed_count", 0ull) < minimum) {
              std::this_thread::sleep_for(child_poll);
              continue;
            }
            return value;
          } catch (const std::exception &exception) {
            // Atomic publication means a parse error is not a partial write. Keep the last
            // error briefly so transient antivirus/file-sharing races do not kill a job.
            last_parse_error = exception.what();
          }
        }
        if (!child.running()) {
          throw worker_error(
            std::string(description) + " exited before acknowledging frame " +
            std::to_string(minimum) + " (exit " +
            std::to_string(child.exit_code()) + ")" +
            (last_parse_error.empty() ? "" : ": " + last_parse_error)
          );
        }
        std::this_thread::sleep_for(child_poll);
      }
      throw worker_error(
        "timed out waiting for " + std::string(description) +
        (last_parse_error.empty() ? "" : ": " + last_parse_error)
      );
    }

    void publish_producer_done(
      const fs::path &directory,
      const std::uint64_t frame_count
    ) {
      write_json_atomic(directory / ".producer-done.json", {
                                                             {"schema", 1},
                                                             {"status", "complete"},
                                                             {"frame_count", frame_count},
                                                           });
    }

    void publish_producer_failed(
      const fs::path &directory,
      const std::string &error
    ) noexcept {
      try {
        write_json_atomic(directory / ".producer-failed.json", {
                                                                 {"schema", 1},
                                                                 {"status", "failed"},
                                                                 {"error", error.substr(0, 1024)},
                                                               });
      } catch (...) {
      }
    }

    class streaming_decoder_t {
    public:
      streaming_decoder_t(
        const worker_spec_t &spec,
        const media_contract_t &media,
        const fs::path &log_path
      ):
          media_(media),
          format_(media.color == media_color_e::sdr ? "bmp" : "pfm"),
          process_(child_process_t::launch(build_decoder_command(spec, media), spec.sunshine_executable.parent_path(), log_path, std::nullopt, true)) {
      }

      [[nodiscard]] const std::string &format() const {
        return format_;
      }

      fs::path publish_next(
        const fs::path &directory,
        const std::uint64_t sequence
      ) {
        if (sequence != next_sequence_ || sequence == 0 || sequence > media_.frames.size()) {
          throw worker_error(
            "streaming decoder sequence mismatch: got " +
            std::to_string(sequence) + ", expected " +
            std::to_string(next_sequence_)
          );
        }
        std::error_code ec;
        fs::create_directories(directory, ec);
        if (ec) {
          throw worker_error("cannot create native follow directory");
        }
        const auto output =
          directory / ("frame_" + frame_id(sequence) + "." + format_);
        if (fs::exists(output, ec) || ec) {
          throw worker_error("refusing to replace a follow frame");
        }
        if (format_ == "bmp") {
          write_bmp(output);
        } else {
          write_pfm(output);
        }
        ++next_sequence_;
        return output;
      }

      void finish() {
        if (next_sequence_ != media_.frames.size() + 1) {
          throw worker_error("streaming decoder finished at the wrong frame count");
        }
        // Drain detection and process completion must be observed together. Waiting for exit
        // first deadlocks if FFmpeg has an unexpected extra frame blocked by the bounded pipe.
        const auto deadline = std::chrono::steady_clock::now() + 30s;
        while (process_.running() &&
               std::chrono::steady_clock::now() < deadline) {
          if (process_.read_one_if_available()) {
            process_.terminate();
            throw worker_error(
              "streaming decoder emitted more frames than FFprobe reported"
            );
          }
          std::this_thread::sleep_for(child_poll);
        }
        if (process_.running()) {
          process_.terminate();
          throw worker_error(
            "streaming decoder did not terminate at the probed frame count"
          );
        }
        const int code = process_.wait(1s);
        if (code != 0) {
          throw worker_error(
            "streaming decoder exited " + std::to_string(code)
          );
        }
        if (process_.read_one_if_available()) {
          throw worker_error(
            "streaming decoder emitted more frames than FFprobe reported"
          );
        }
        process_.terminate();
      }

      void abort() noexcept {
        process_.terminate();
      }

    private:
      void write_bmp(const fs::path &path) {
        const auto pixels_size =
          static_cast<std::uint64_t>(media_.width) *
          media_.height * 4ull;
        if (pixels_size > std::numeric_limits<std::size_t>::max() || pixels_size + 54ull > std::numeric_limits<std::uint32_t>::max()) {
          throw worker_error("source is too large for the BMP follow contract");
        }
        std::vector<std::uint8_t> bytes(
          static_cast<std::size_t>(54ull + pixels_size),
          0
        );
        const auto write16 = [&](const std::size_t offset, const std::uint16_t value) {
          bytes[offset] = static_cast<std::uint8_t>(value);
          bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
        };
        const auto write32 = [&](const std::size_t offset, const std::uint32_t value) {
          for (std::size_t index = 0; index < 4; ++index) {
            bytes[offset + index] =
              static_cast<std::uint8_t>(value >> (8 * index));
          }
        };
        bytes[0] = 'B';
        bytes[1] = 'M';
        write32(2, static_cast<std::uint32_t>(bytes.size()));
        write32(10, 54);
        write32(14, 40);
        write32(18, media_.width);
        write32(
          22,
          std::bit_cast<std::uint32_t>(
            -static_cast<std::int32_t>(media_.height)
          )
        );
        write16(26, 1);
        write16(28, 32);
        write32(34, static_cast<std::uint32_t>(pixels_size));
        process_.read_exact(bytes.data() + 54, static_cast<std::size_t>(pixels_size));
        write_bytes_atomic(path, bytes);
      }

      void write_pfm(const fs::path &path) {
        const auto magic = process_.read_line(16);
        const auto dimensions = process_.read_line(128);
        const auto scale = process_.read_line(64);
        if (magic != "PF\n" && magic != "PF\r\n") {
          throw worker_error("FFmpeg PFM stream is not RGB float32");
        }
        std::istringstream dimension_stream(dimensions);
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::string trailing;
        if (!(dimension_stream >> width >> height) || (dimension_stream >> trailing) || width != media_.width || height != media_.height) {
          throw worker_error("FFmpeg PFM dimensions changed during the clip");
        }
        char *end = nullptr;
        const float scale_value = std::strtof(scale.c_str(), &end);
        while (end && *end &&
               std::isspace(static_cast<unsigned char>(*end))) {
          ++end;
        }
        if (!end || *end || scale_value != -1.0f) {
          throw worker_error(
            "FFmpeg PFM must be little-endian float32 with scale -1"
          );
        }
        const auto floats =
          static_cast<std::uint64_t>(width) * height * 3ull;
        if (floats > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
          throw worker_error("PFM frame is too large");
        }
        std::vector<float> payload(static_cast<std::size_t>(floats));
        process_.read_exact(payload.data(), payload.size() * sizeof(float));
        if (!std::all_of(payload.begin(), payload.end(), [](const float value) {
              return std::isfinite(value) && std::abs(value) <= 65504.0f;
            })) {
          throw worker_error("FFmpeg emitted a non-finite/out-of-range PFM");
        }
        std::vector<std::uint8_t> bytes;
        const std::string header = magic + dimensions + scale;
        bytes.resize(header.size() + payload.size() * sizeof(float));
        std::memcpy(bytes.data(), header.data(), header.size());
        std::memcpy(
          bytes.data() + header.size(),
          payload.data(),
          payload.size() * sizeof(float)
        );
        write_bytes_atomic(path, bytes);
      }

      const media_contract_t &media_;
      std::string format_;
      child_process_t process_;
      std::uint64_t next_sequence_ = 1;
    };

    bool adaptive_trace_flags_valid(
      const float cut_flags,
      const std::uint32_t analysis_flags
    ) {
      return
        std::isfinite(cut_flags) &&
        cut_flags >= 0.0f &&
        cut_flags <=
          static_cast<float>(sbs_adaptive_state::known_cut_flag_mask) &&
        std::trunc(cut_flags) == cut_flags &&
        (
          analysis_flags & ~sbs_adaptive_state::known_analysis_flag_mask
        ) == 0u;
    }

    template<std::size_t Size>
    bool json_object_has_exact_keys(
      const nlohmann::json &value,
      const std::array<std::string_view, Size> &keys
    ) {
      if (!value.is_object() || value.size() != keys.size()) {
        return false;
      }
      return std::ranges::all_of(keys, [&](const std::string_view key) {
        return value.contains(std::string {key});
      });
    }

    scene_frame_t parse_trace_frame(
      const nlohmann::json &value,
      const frame_timing_t &timing,
      const rational_t time_base,
      const std::uint64_t cache_bytes
    ) {
      using sbs_adaptive_state::word_e;
      if (!json_object_has_exact_keys(value, sbs_adaptive_state::frame_keys) ||
          value.value("record", "") != "frame" ||
          value.value("frame_id", "") != frame_id(timing.sequence) ||
          value.value("source_index", max_sequence) != timing.sequence - 1 ||
          !value["values"].is_array() ||
          value["values"].size() != sbs_adaptive_state::word_count) {
        throw worker_error("adaptive trace frame identity/schema mismatch");
      }
      const auto &words = value["values"];
      const auto is_uint32 = [](const nlohmann::json &encoded) {
        return encoded.is_number_unsigned() &&
               encoded.get<std::uint64_t>() <=
                 std::numeric_limits<std::uint32_t>::max();
      };
      for (const auto &field : sbs_adaptive_state::fields) {
        const auto &encoded = words[sbs_adaptive_state::index(field.word)];
        if (field.gpu_encoding == sbs_adaptive_state::gpu_encoding_e::uint_bits ||
            field.gpu_encoding ==
              sbs_adaptive_state::gpu_encoding_e::uint_valued_float) {
          if (!is_uint32(encoded)) {
            throw worker_error("adaptive trace integer field encoding mismatch");
          }
          continue;
        }
        if (!encoded.is_number()) {
          throw worker_error("adaptive trace float field encoding mismatch");
        }
        const auto decoded = encoded.get<double>();
        if (!std::isfinite(decoded) ||
            std::abs(decoded) > std::numeric_limits<float>::max()) {
          throw worker_error("adaptive trace contains a non-finite float field");
        }
      }
      for (const auto word : sbs_adaptive_state::reserved_words) {
        const auto index = sbs_adaptive_state::index(word);
        const auto &field = sbs_adaptive_state::fields[index];
        const auto matches =
          field.gpu_encoding == sbs_adaptive_state::gpu_encoding_e::uint_bits ?
            words[index].get<std::uint32_t>() ==
              sbs_adaptive_state::initial_words[index] :
            words[index].get<float>() == field.initial_value;
        if (!matches) {
          throw worker_error("adaptive trace reserved field is non-default");
        }
      }
      const auto &cut_contract_tag = words[sbs_adaptive_state::index(
        word_e::cut_contract_tag_bits
      )];
      if (!is_uint32(cut_contract_tag) ||
          cut_contract_tag.get<std::uint32_t>() !=
            sbs_adaptive_state::cut_contract_tag) {
        throw worker_error("adaptive trace cut contract tag mismatch");
      }
      const auto scalar = [&](const word_e word) -> std::optional<float> {
        const auto index = sbs_adaptive_state::index(word);
        if (words[index].is_null()) {
          return std::nullopt;
        }
        const auto result = words[index].get<float>();
        if (!std::isfinite(result) || result < 0.0f) {
          return std::nullopt;
        }
        return result;
      };
      scene_frame_t frame;
      frame.sequence = timing.sequence;
      frame.frame_id = frame_id(timing.sequence);
      frame.depth_updated = value.at("depth_updated").get<bool>();
      if (!frame.depth_updated) {
        throw worker_error("adaptive trace uses the retired depth-reuse cadence");
      }
      frame.depth_change_fraction = scalar(word_e::current_depth_change_fraction);
      frame.raw_rgb_change_fraction = scalar(word_e::raw_rgb_change_fraction);
      frame.structural_change_fraction = scalar(word_e::structural_change_fraction);
      frame.current_structural_support_fraction =
        scalar(word_e::current_structural_support_fraction);
      frame.previous_structural_support_fraction =
        scalar(word_e::previous_structural_support_fraction);
      frame.common_structural_support_fraction =
        scalar(word_e::common_structural_support_fraction);
      const auto cut_flags_value =
        words[sbs_adaptive_state::index(word_e::cut_flags)].get<float>();
      const auto analysis_flags =
        words[sbs_adaptive_state::index(word_e::analysis_flags)]
          .get<std::uint32_t>();
      if (!adaptive_trace_flags_valid(cut_flags_value, analysis_flags)) {
        throw worker_error("adaptive trace contains invalid or unknown flags");
      }
      const auto cut_flags = static_cast<std::uint32_t>(cut_flags_value);
      const auto require_boolean_duplicate = [&](const char *name, const bool expected) {
        if (!value.at(name).is_boolean() || value.at(name).get<bool>() != expected) {
          throw worker_error(
            std::string {"adaptive trace duplicate disagrees for "} + name
          );
        }
      };
      require_boolean_duplicate(
        "geometry_armed",
        (cut_flags & sbs_adaptive_state::cut_flag_geometry_armed) != 0u
      );
      require_boolean_duplicate(
        "appearance_armed",
        (cut_flags & sbs_adaptive_state::cut_flag_appearance_armed) != 0u
      );
      require_boolean_duplicate(
        "geometry_low_once",
        (cut_flags & sbs_adaptive_state::cut_flag_geometry_low_once) != 0u
      );
      require_boolean_duplicate(
        "appearance_quiet_once",
        (cut_flags & sbs_adaptive_state::cut_flag_appearance_quiet_once) != 0u
      );
      require_boolean_duplicate(
        "cut_latched",
        (cut_flags & sbs_adaptive_state::cut_flag_latched) != 0u
      );
      require_boolean_duplicate(
        "appearance_recovery",
        (cut_flags & sbs_adaptive_state::cut_flag_appearance_recovery) != 0u
      );
      require_boolean_duplicate(
        "geometry_confirmation_pending",
        (cut_flags &
         sbs_adaptive_state::cut_flag_geometry_confirmation_pending) != 0u
      );
      const auto pulse_value = words[sbs_adaptive_state::index(
        word_e::hard_cut_pulse
      )].get<float>();
      if (pulse_value != 0.0f && pulse_value != 1.0f) {
        throw worker_error("adaptive trace hard-cut pulse is not exactly 0 or 1");
      }
      frame.hard_cut_pulse = pulse_value > 0.5f;
      require_boolean_duplicate("hard_cut_pulse", frame.hard_cut_pulse);
      for (const auto &[name, word] : std::array {
             std::pair {"hard_cut_count", word_e::hard_cut_count},
             std::pair {"empty_raw_count", word_e::empty_raw_count},
             std::pair {"collapsed_raw_count", word_e::collapsed_raw_count},
           }) {
        const auto expected = words[sbs_adaptive_state::index(word)].get<std::uint32_t>();
        if (expected > sbs_adaptive_state::counter_max ||
            !is_uint32(value.at(name)) ||
            value.at(name).get<std::uint32_t>() != expected) {
          throw worker_error(
            std::string {"adaptive trace counter duplicate disagrees for "} + name
          );
        }
      }
      frame.analysis_flags = analysis_flags;
      frame.pts_seconds = time_base.seconds(timing.pts);
      frame.duration_seconds = time_base.seconds(timing.duration);
      frame.cache_bytes = cache_bytes;
      return frame;
    }

    class trace_tail_t {
    public:
      explicit trace_tail_t(
        fs::path path,
        const bool bounded_snapshot = false
      ):
          path_(std::move(path)),
          bounded_snapshot_(bounded_snapshot) {
      }

      nlohmann::json read_header(const child_process_t &child) {
        nlohmann::json header;
        if (bounded_snapshot_) {
          header = read_snapshot(child, "adaptive_state_header.json");
        } else {
          ensure_open(child);
          header = parse_line(read_complete_line(child));
        }
        if (!json_object_has_exact_keys(header, sbs_adaptive_state::header_keys) ||
            header.value("record", "") != "header" ||
            !header["schema"].is_number_unsigned() ||
            header["schema"].get<std::uint64_t>() !=
              sbs_adaptive_state::schema_version ||
            !header["fields"].is_array() ||
            header["fields"].size() != sbs_adaptive_state::word_count) {
          throw worker_error("adaptive trace header has the wrong schema or word count");
        }
        for (std::size_t index = 0; index < sbs_adaptive_state::fields.size(); ++index) {
          const auto &expected = sbs_adaptive_state::fields[index];
          const auto &field = header["fields"][index];
          if (!field.is_object() || field.size() != 2u ||
              field.value("name", "") != expected.name ||
              field.value("type", "") != expected.json_type) {
            throw worker_error(
              "adaptive trace field layout differs at word " +
              std::to_string(index)
            );
          }
        }
        nlohmann::json expected_flags = nlohmann::json::object();
        for (const auto &flag : sbs_adaptive_state::analysis_flag_bits) {
          expected_flags[std::string {flag.name}] = flag.bit;
        }
        if (header.value("source", "") != sbs_adaptive_state::source ||
            header.value("capture", "") != sbs_adaptive_state::capture ||
            !header.contains("analysis_flag_bits") ||
            header["analysis_flag_bits"] != expected_flags) {
          throw worker_error("adaptive trace attribution differs");
        }
        if (!header.contains("config") || !header["config"].is_object() ||
            header["config"].size() != sbs_adaptive_state::config_keys.size()) {
          throw worker_error("adaptive trace config contract differs");
        }
        const auto &config = header["config"];
        for (const auto key : sbs_adaptive_state::config_keys) {
          if (!config.contains(std::string {key})) {
            throw worker_error("adaptive trace config contract differs");
          }
        }
        const auto model = config.value("model", "");
        const auto pop_strength = config.value(
          "pop_strength", std::numeric_limits<double>::quiet_NaN()
        );
        const auto depth_reuse_interval = config.value("depth_reuse_interval", 0);
        if (model.empty() || !std::isfinite(pop_strength) ||
            pop_strength < 0.25 || pop_strength > 2.0 ||
            depth_reuse_interval != 1) {
          throw worker_error("adaptive trace config values are unsupported");
        }
        return header;
      }

      nlohmann::json read_frame(
        const child_process_t &child,
        const std::uint64_t sequence
      ) {
        auto value = bounded_snapshot_ ?
                       read_snapshot(child, "adaptive_state_frame.json") :
                       parse_line(read_complete_line(child));
        if (value.value("record", "") != "frame" || value.value("source_index", max_sequence) != sequence - 1) {
          throw worker_error("adaptive trace is not contiguous");
        }
        ++frame_count_;
        return value;
      }

      void finish(const std::uint64_t expected) {
        if (frame_count_ != expected) {
          throw worker_error("adaptive trace frame count mismatch");
        }
        if (bounded_snapshot_) {
          std::error_code ec;
          if (
            fs::exists(path_ / "adaptive_state_header.json", ec) ||
            ec ||
            fs::exists(path_ / "adaptive_state_frame.json", ec) ||
            ec
          ) {
            throw worker_error(
              "bounded adaptive trace retained a consumed snapshot"
            );
          }
          return;
        }
        stream_.clear();
        char trailing = 0;
        if (stream_.get(trailing)) {
          throw worker_error("adaptive trace has records past the source timeline");
        }
      }

    private:
      nlohmann::json read_snapshot(
        const child_process_t &child,
        const std::string_view filename
      ) {
        constexpr std::uintmax_t max_snapshot_bytes = 1ull * 1024ull * 1024ull;
        const auto snapshot = path_ / filename;
        const auto deadline = std::chrono::steady_clock::now() + child_timeout;
        while (std::chrono::steady_clock::now() < deadline) {
          std::error_code ec;
          if (fs::is_regular_file(snapshot, ec) && !ec) {
            try {
              const auto bytes =
                read_bounded_bytes(snapshot, max_snapshot_bytes);
              auto value = parse_json_without_duplicate_keys(bytes);
              remove_file_checked(snapshot);
              return value;
            } catch (const worker_error &) {
              throw;
            } catch (const std::exception &exception) {
              throw worker_error(
                "invalid bounded adaptive trace JSON: " +
                std::string(exception.what())
              );
            }
          }
          if (ec) {
            throw worker_error(
              "cannot inspect bounded adaptive trace snapshot"
            );
          }
          if (!child.running()) {
            throw worker_error(
              "analysis exited before publishing the expected trace snapshot"
            );
          }
          std::this_thread::sleep_for(child_poll);
        }
        throw worker_error("timed out reading bounded adaptive trace");
      }

      void ensure_open(const child_process_t &child) {
        const auto deadline = std::chrono::steady_clock::now() + child_timeout;
        while (std::chrono::steady_clock::now() < deadline) {
          stream_.open(path_, std::ios::binary);
          if (stream_) {
            return;
          }
          stream_.clear();
          if (!child.running()) {
            throw worker_error("analysis exited before publishing its trace");
          }
          std::this_thread::sleep_for(child_poll);
        }
        throw worker_error("timed out opening adaptive trace");
      }

      std::string read_complete_line(const child_process_t &child) {
        const auto deadline = std::chrono::steady_clock::now() + child_timeout;
        while (std::chrono::steady_clock::now() < deadline) {
          const auto position = stream_.tellg();
          std::string line;
          if (std::getline(stream_, line)) {
            if (line.size() > 1024 * 1024) {
              throw worker_error("adaptive trace record is unreasonably large");
            }
            return line;
          }
          stream_.clear();
          stream_.seekg(position);
          if (!child.running()) {
            throw worker_error("analysis exited before publishing the expected trace");
          }
          std::this_thread::sleep_for(child_poll);
        }
        throw worker_error("timed out tailing adaptive trace");
      }

      nlohmann::json parse_line(const std::string &line) {
        try {
          return parse_json_without_duplicate_keys(line);
        } catch (const std::exception &exception) {
          throw worker_error(
            std::string {"invalid adaptive trace JSON: "} + exception.what()
          );
        }
      }

      fs::path path_;
      bool bounded_snapshot_ = false;
      std::ifstream stream_;
      std::uint64_t frame_count_ = 0;
    };

    struct cache_contract_t {
      std::uint64_t processed_count = 0;
      std::uint64_t depth_bytes = 0;
      std::uint64_t state_bytes = 0;
      std::uint32_t source_width = 0;
      std::uint32_t source_height = 0;
      std::uint32_t sbs_width = 0;
      std::uint32_t sbs_height = 0;
      std::string extension;
      nlohmann::json value;
    };

    cache_contract_t parse_cache_contract(
      const fs::path &cache_directory,
      const std::uint64_t sequence,
      const media_contract_t &media
    ) {
      const auto value = read_json(
        cache_directory / "scene_cache_contract.json"
      );
      if (value.value("schema", 0u) != scene_cache_contract_schema || value.value("status", "") != "running" || value.value("first_sequence", 0) != 1 || value.value("processed_count", 0ull) != sequence || !value.value("atomic_frame_publication", false)) {
        throw worker_error("running scene-cache sequence contract mismatch");
      }
      const auto &source = value.at("source");
      const auto &depth = value.at("depth");
      const auto &state = value.at("state");
      const auto &render = value.at("render_config");
      const auto &packed = value.at("packed_sbs");
      cache_contract_t result;
      result.processed_count = sequence;
      result.depth_bytes = depth.at("bytes_per_frame").get<std::uint64_t>();
      result.state_bytes = state.at("bytes_per_frame").get<std::uint64_t>();
      result.source_width = source.at("width").get<std::uint32_t>();
      result.source_height = source.at("height").get<std::uint32_t>();
      result.sbs_width = packed.at("width").get<std::uint32_t>();
      result.sbs_height = packed.at("height").get<std::uint32_t>();
      result.extension = packed.at("file_extension").get<std::string>();
      result.value = value;
      const std::string expected_frame_format =
        media.color == media_color_e::sdr ?
          "sRGB-BMP-WIC" :
          "linear-scRGB-f32-pfm";
      const auto depth_width = depth.at("width").get<std::uint32_t>();
      const auto depth_height = depth.at("height").get<std::uint32_t>();
      const auto expected_depth_bytes =
        static_cast<std::uint64_t>(depth_width) * depth_height * sizeof(float);
      if (result.source_width != media.width || result.source_height != media.height || source.at("frame_format").get<std::string>() != expected_frame_format || result.sbs_width == 0 || result.sbs_height == 0 || result.sbs_width % 2 != 0 || packed.at("eye_width").get<std::uint32_t>() * 2 != result.sbs_width || packed.at("eye_height").get<std::uint32_t>() != result.sbs_height || !packed.at("atomic_replay_publication").get<bool>() || result.extension != (media.color == media_color_e::sdr ? "png" : "pfm") || depth_width == 0u || depth_height == 0u || result.depth_bytes != expected_depth_bytes || depth.at("dtype").get<std::string>() != "float32-le" || depth.at("dxgi_format").get<std::string>() != "R32_FLOAT" || depth.at("semantics").get<std::string>() != "depth-coordinate-v2-signed-final-parallax-source-u" || state.at("schema").get<int>() != 2 || state.at("contract_schema").get<std::uint32_t>() != models::depth_coordinate_v2::contract_schema || state.at("contract_tag").get<std::uint32_t>() != models::depth_coordinate_v2::contract_tag || state.at("word_count").get<std::size_t>() != models::depth_coordinate_v2::state_words_t {}.size() || result.state_bytes != sizeof(models::depth_coordinate_v2::state_words_t) || render.at("renderer").get<std::string>() != "depth-coordinate-v2-live-signed-parallax" || render.at("producer_source_closure_sha256").get<std::string>() != models::depth_coordinate_v2::shader_source_closure_sha256 || render.at("renderer_source_closure_sha256").get<std::string>() != models::host_sbs_shader_cache::parallax_v2_live_renderer_source_closure_sha256) {
        throw worker_error("scene-cache media/layout contract mismatch");
      }
      const auto stem = "frame_" + frame_id(sequence);
      const auto depth_path = cache_directory / (stem + ".depth.r32f");
      const auto state_path = cache_directory / (stem + ".state.u32");
      std::error_code ec;
      if (!fs::is_regular_file(depth_path, ec) || ec || fs::file_size(depth_path, ec) != result.depth_bytes || ec || !fs::is_regular_file(state_path, ec) || ec || fs::file_size(state_path, ec) != result.state_bytes || ec) {
        throw worker_error("scene-cache ACK lacks its exact depth/state pair");
      }
      return result;
    }

    nlohmann::json scene_plan_json(const scene_plan_t &scene) {
      return {
        {"schema", 2},
        {"version", "scene-plan-v2"},
        {"cache_contract_schema", scene_cache_contract_schema},
        {"scenes", nlohmann::json::array({
                     {
                       {"start_sequence", scene.start_sequence},
                       {"end_sequence_exclusive", scene.end_sequence_exclusive},
                     },
                   })},
      };
    }

    nlohmann::json boundary_json(const boundary_audit_t &boundary) {
      return {
        {"proposal_sequences", boundary.proposal_sequences},
        {"proposal_frame_ids", boundary.proposal_frame_ids},
        {"final_sequence", boundary.final_sequence},
        {"decision", boundary_decision_name(boundary.decision)},
        {"reason", boundary.reason},
        {"accepted", boundary.accepted},
        {"semantic_cut", boundary.semantic_cut},
        {"truncated", boundary.truncated},
        {"budget_forced", boundary.budget_forced},
        {"revision_depth_updates", boundary.revision_depth_updates},
        {"revision_source_frames", boundary.revision_source_frames},
        {"candidate_count", boundary.candidate_count},
        {"selected_evidence_score", boundary.selected_evidence_score},
        {"evidence_window_first_sequence",
         boundary.evidence_window_first_sequence},
        {"evidence_window_last_sequence",
         boundary.evidence_window_last_sequence},
        {"selected_sequence", boundary.selected_sequence},
        {"selected_frame_id", boundary.selected_frame_id},
        {"selected_depth_change_fraction",
         boundary.selected_depth_change_fraction},
        {"selected_raw_rgb_change_fraction",
         boundary.selected_raw_rgb_change_fraction},
        {"selected_structural_change_fraction",
         boundary.selected_structural_change_fraction},
        {"selected_appearance_qualified",
         boundary.selected_appearance_qualified},
        {"selected_geometry_qualified",
         boundary.selected_geometry_qualified},
        {"selected_relative_geometry_spike",
         boundary.selected_relative_geometry_spike},
      };
    }

    nlohmann::json scene_evidence_json(const scene_evidence_t &evidence) {
      return {
        {"source_frame_count", evidence.source_frame_count},
        {"depth_update_count", evidence.depth_update_count},
        {"appearance_veto_count", evidence.appearance_veto_count},
        {"depth_change_max", evidence.depth_change_max},
      };
    }

    nlohmann::json scene_json(const scene_plan_t &scene) {
      return {
        {"scene_id", scene.scene_id},
        {"semantic_scene_id", scene.semantic_scene_id},
        {"start_sequence", scene.start_sequence},
        {"end_sequence_exclusive", scene.end_sequence_exclusive},
        {"frame_count", scene.frame_count},
        {"cache_bytes", scene.cache_bytes},
        {"start_pts_seconds", scene.start_pts_seconds},
        {"end_pts_seconds_exclusive", scene.end_pts_seconds_exclusive},
        {"evidence", scene_evidence_json(scene.evidence)},
        {"boundary", boundary_json(scene.boundary)},
        {"ground_truth", scene.ground_truth},
        {"cut_state_semantics", scene.cut_state_semantics},
        {"known_limit", scene.known_limit},
      };
    }

    nlohmann::json scene_progress_json(const scene_plan_t &scene) {
      return {
        {"scene_id", scene.scene_id},
        {"semantic_scene_id", scene.semantic_scene_id},
        {"start_sequence", scene.start_sequence},
        {"end_sequence_exclusive", scene.end_sequence_exclusive},
        {"frame_count", scene.frame_count},
        {"boundary", {
                       {"decision", boundary_decision_name(scene.boundary.decision)},
                       {"final_sequence", scene.boundary.final_sequence},
                       {"accepted", scene.boundary.accepted},
                       {"semantic_cut", scene.boundary.semantic_cut},
                       {"budget_forced", scene.boundary.budget_forced},
                       {"revision_depth_updates", scene.boundary.revision_depth_updates},
                       {"revision_source_frames", scene.boundary.revision_source_frames},
                     }},
      };
    }

    void release_cache_scene(
      const fs::path &cache,
      const scene_plan_t &scene
    ) {
      for (std::uint64_t sequence = scene.start_sequence;
           sequence < scene.end_sequence_exclusive;
           ++sequence) {
        const auto stem = "frame_" + frame_id(sequence);
        remove_file_checked(cache / (stem + ".depth.r32f"));
        remove_file_checked(cache / (stem + ".state.u32"));
      }
    }

    std::string codec_name_for_request(const std::string &codec) {
      if (codec == "hevc_nvenc") {
        return "hevc";
      }
      if (codec == "av1_nvenc") {
        return "av1";
      }
      throw worker_error("unsupported offline codec");
    }

    std::vector<std::string> codec_arguments_impl(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const bool hdr,
      const std::uint32_t encoded_width,
      const std::uint32_t encoded_height
    ) {
      if (spec.codec == "hevc_nvenc") {
        std::vector<std::string> result {
          "-c:v",
          "hevc_nvenc",
          "-preset",
          "p5",
          "-tune",
          "hq",
          "-rc",
          "vbr",
          "-cq",
          "18",
          "-b:v",
          "0",
          "-bf",
          "0",
          "-pix_fmt",
          hdr ? "p010le" : "yuv420p",
        };
        if (hdr) {
          result.insert(
            result.end(),
            {"-profile:v", "main10", "-extra_sei", "1"}
          );
        }
        return result;
      }
      if (spec.codec == "av1_nvenc") {
        std::vector<std::string> result {
          "-c:v",
          "av1_nvenc",
          "-preset",
          "p5",
          "-tune",
          "hq",
          "-rc",
          "vbr",
          "-cq",
          "20",
          "-b:v",
          "0",
          "-bf",
          "0",
          "-pix_fmt",
          hdr ? "p010le" : "yuv420p",
          "-level",
          select_av1_level(media, encoded_width, encoded_height),
        };
        if (hdr) {
          result.insert(result.end(), {"-extra_sei", "1"});
        }
        return result;
      }
      throw worker_error("unsupported offline codec");
    }

    void write_whole_clip_concat(
      const fs::path &path,
      const scene_frame_server_t &frame_server,
      const media_contract_t &media
    ) {
      std::ostringstream contents;
      contents.imbue(std::locale::classic());
      contents << "ffconcat version 1.0\n";
      for (const auto &timing : media.frames) {
        const auto sequence = timing.sequence;
        contents << "file '" << frame_server.frame_url(sequence) << "'\n"
                 // Image timestamps are deliberately synthetic. The filter script below
                 // replaces every PTS from source frame N in the source rational time base.
                 << "option framerate 1/1\n"
                 << "duration 1\n";
      }
      write_text_atomic(path, contents.str());
    }

    struct pts_run_t {
      std::size_t begin = 0;
      std::size_t end = 0;
      std::int64_t first_pts = 0;
      std::int64_t step = 0;
    };

    std::vector<pts_run_t> pts_runs(const media_contract_t &media) {
      std::vector<pts_run_t> runs;
      if (media.frames.empty()) {
        return runs;
      }
      std::size_t begin = 0;
      while (begin < media.frames.size()) {
        const auto step = media.frames[begin].duration;
        std::size_t end = begin + 1;
        while (end < media.frames.size() &&
               media.frames[end].pts - media.frames[end - 1].pts == step) {
          ++end;
        }
        runs.push_back({
          .begin = begin,
          .end = end,
          .first_pts = media.frames[begin].pts,
          .step = step,
        });
        begin = end;
      }
      return runs;
    }

    void append_pts_lookup(
      std::ostringstream &output,
      const std::vector<pts_run_t> &runs,
      const std::size_t begin,
      const std::size_t end
    ) {
      if (end <= begin || end > runs.size()) {
        throw worker_error("cannot build an empty/out-of-range PTS lookup");
      }
      if (end - begin == 1) {
        const auto &run = runs[begin];
        if (run.end - run.begin == 1) {
          output << run.first_pts;
        } else {
          output << '(' << run.first_pts << "+(N-" << run.begin
                 << ")*" << run.step << ')';
        }
        return;
      }
      const auto middle = begin + (end - begin) / 2;
      output << "if(lt(N\\," << runs[middle].begin << ")\\,";
      append_pts_lookup(output, runs, begin, middle);
      output << "\\,";
      append_pts_lookup(output, runs, middle, end);
      output << ')';
    }

    void write_whole_clip_filter(
      const fs::path &path,
      const media_contract_t &media,
      const bool hdr,
      const std::uint32_t sbs_width,
      const std::uint32_t sbs_height
    ) {
      const auto runs = pts_runs(media);
      if (runs.empty()) {
        throw worker_error("cannot encode an empty source timeline");
      }
      constexpr std::size_t max_pts_runs = 1000000;
      if (runs.size() > max_pts_runs) {
        throw worker_error(
          "source VFR timeline exceeds one million exact timing runs"
        );
      }
      std::ostringstream lookup;
      lookup.imbue(std::locale::classic());
      append_pts_lookup(lookup, runs, 0, runs.size());
      const auto expression = lookup.str();
      constexpr std::size_t max_filter_script_bytes = 128ull * 1024ull * 1024ull;
      if (expression.size() > max_filter_script_bytes / (hdr ? 2 : 1)) {
        throw worker_error(
          "exact source timeline requires an unreasonably large PTS map"
        );
      }
      const auto timing_filter =
        "settb=expr=" + std::to_string(media.time_base.numerator) + "/" +
        std::to_string(media.time_base.denominator) + ",setpts='" +
        expression + "'";
      std::ostringstream graph;
      graph.imbue(std::locale::classic());
      if (hdr) {
        graph
          << "[0:v:0]trim=end_frame=" << media.frames.size() << ','
          << timing_filter << ',' << hdr_decode_filter(media)
          << ",scale=" << sbs_width << ':' << sbs_height
          // Scaling the source to the packed SBS width would otherwise preserve
          // its original display aspect by changing the donor SAR to 1:2.
          << ":flags=bilinear,setsar=1/1[donor];\n"
          << "[1:v:0]" << timing_filter
          << ",format=gbrpf32le[sbs];\n"
          << "[donor][sbs]blend=all_expr=B:shortest=1,"
          << hdr_encode_filter(media) << "[vout]\n";
      } else {
        graph << "[0:v:0]" << timing_filter << "[vout]\n";
      }
      const auto contents = graph.str();
      if (contents.size() > max_filter_script_bytes) {
        throw worker_error(
          "exact source timeline filter exceeds the 128 MiB safety limit"
        );
      }
      write_text_atomic(path, contents);
    }

    std::vector<std::string> build_whole_clip_encode_command(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const fs::path &concat,
      const fs::path &filter_script,
      const fs::path &encoded_video,
      const std::uint32_t encoded_width,
      const std::uint32_t encoded_height
    ) {
      if (media.time_base.denominator > std::numeric_limits<std::int32_t>::max()) {
        throw worker_error(
          "source time-base denominator exceeds the MP4 timescale limit"
        );
      }
      const bool hdr = media.color != media_color_e::sdr;
      std::vector<std::string> command {
        path_utf8(spec.ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-xerror",
        "-nostdin",
        "-y",
        "-copyts",
      };
      if (hdr) {
        // The original stream is a frame-side-data donor only. Pixel values always come
        // from the replay PFM (`blend ... =B`).
        command.insert(command.end(), {
                                        "-protocol_whitelist",
                                        "file",
                                        "-i",
                                        path_utf8(spec.input_path),
                                      });
      }
      command.insert(command.end(), {
                                      "-protocol_whitelist",
                                      "file,http,tcp",
                                      "-f",
                                      "concat",
                                      "-safe",
                                      "0",
                                      "-i",
                                      path_utf8(concat),
                                    });
      command.insert(command.end(), {
                                      "-filter_complex_script",
                                      path_utf8(filter_script),
                                      "-map",
                                      "[vout]",
                                      "-an",
                                      "-sn",
                                      "-dn",
                                      "-fps_mode",
                                      "passthrough",
                                      "-enc_time_base",
                                      "filter",
                                    });
      const auto codec =
        build_codec_arguments(spec, media, hdr, encoded_width, encoded_height);
      command.insert(command.end(), codec.begin(), codec.end());
      command.insert(command.end(), {
                                      "-bsf:v",
                                      "setts=duration=if(eq(N\\," + std::to_string(media.frames.size() - 1) + ")\\," + std::to_string(media.frames.back().duration) + "\\,DURATION)",
                                      "-avoid_negative_ts",
                                      "disabled",
                                    });
      if (!media.color_range.empty()) {
        command.insert(command.end(), {
                                        "-color_range",
                                        media.color_range,
                                      });
      }
      if (!media.color_space.empty()) {
        command.insert(command.end(), {
                                        "-colorspace",
                                        media.color_space,
                                      });
      }
      if (!media.color_transfer.empty()) {
        command.insert(command.end(), {
                                        "-color_trc",
                                        media.color_transfer,
                                      });
      }
      if (!media.color_primaries.empty()) {
        command.insert(command.end(), {
                                        "-color_primaries",
                                        media.color_primaries,
                                      });
      }
      command.insert(command.end(), {
                                      "-video_track_timescale",
                                      std::to_string(media.time_base.denominator),
                                      "-movie_timescale",
                                      std::to_string(media.time_base.denominator),
                                      "-movflags",
                                      "+write_colr",
                                      "-f",
                                      "mp4",
                                      path_utf8(encoded_video),
                                    });
      return command;
    }

    class whole_clip_encoder_t {
    public:
      whole_clip_encoder_t(
        const worker_spec_t &spec,
        const media_contract_t &media,
        const fs::path &work,
        const std::uint32_t sbs_width,
        const std::uint32_t sbs_height
      ):
          frame_server_(
            1,
            static_cast<std::uint64_t>(media.frames.size()) + 1,
            media.color == media_color_e::sdr ? "png" : "pfm"
          ),
          encoded_video_(work / "encoded-video.mp4") {
        const auto concat = work / "encoded-video.ffconcat";
        const auto filter_script = work / "encoded-video-filter.txt";
        write_whole_clip_concat(concat, frame_server_, media);
        write_whole_clip_filter(
          filter_script,
          media,
          media.color != media_color_e::sdr,
          sbs_width,
          sbs_height
        );
        process_ = child_process_t::launch(
          build_whole_clip_encode_command(
            spec,
            media,
            concat,
            filter_script,
            encoded_video_,
            sbs_width,
            sbs_height
          ),
          spec.sunshine_executable.parent_path(),
          work / "logs" / "encode-whole-clip.log"
        );
      }

      whole_clip_encoder_t(const whole_clip_encoder_t &) = delete;
      whole_clip_encoder_t &operator=(const whole_clip_encoder_t &) = delete;

      ~whole_clip_encoder_t() {
        abort("whole-clip encoder destroyed");
      }

      void publish(
        const std::uint64_t sequence,
        const fs::path &frame
      ) {
        if (finished_) {
          throw worker_error("cannot publish to a finished whole-clip encoder");
        }
        frame_server_.publish_and_wait(sequence, frame, process_);
      }

      void finish() {
        if (finished_) {
          throw worker_error("whole-clip encoder was already finished");
        }
        frame_server_.finish();
        const int code = process_.wait(
          std::chrono::duration_cast<std::chrono::milliseconds>(child_timeout)
        );
        if (code != 0) {
          throw worker_error(
            "whole-clip encoder exited " + std::to_string(code)
          );
        }
        process_.terminate();
        finished_ = true;
      }

      void abort(const std::string &reason) noexcept {
        if (finished_) {
          return;
        }
        process_.terminate();
        frame_server_.abort(reason);
        finished_ = true;
      }

      [[nodiscard]] const fs::path &encoded_video() const {
        return encoded_video_;
      }

    private:
      scene_frame_server_t frame_server_;
      child_process_t process_;
      fs::path encoded_video_;
      bool finished_ = false;
    };

    struct streaming_probe_stats_t {
      std::size_t peak_retained_frame_descriptors = 0;
      std::uintmax_t probe_bytes = 0;
    };

    media_contract_t parse_ffprobe_frames_incrementally(
      const nlohmann::json &stream,
      std::istream &input,
      bool require_source_packet_durations,
      streaming_probe_stats_t *stats,
      std::string_view description
    );

#ifdef SUNSHINE_TESTS
    media_contract_t parse_ffprobe_frames_incrementally(
      const nlohmann::json &stream,
      const fs::path &frames_path,
      bool require_source_packet_durations,
      streaming_probe_stats_t *stats
    );
#endif

    nlohmann::json media_contract_json(const media_contract_t &media);

    std::vector<std::string> build_ffprobe_metadata_command(
      const worker_spec_t &spec,
      const fs::path &media_path
    ) {
      return {
        path_utf8(spec.ffprobe_executable),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-show_entries",
        "stream=index,codec_name,profile,codec_tag_string,pix_fmt,width,height,"
        "sample_aspect_ratio,field_order,time_base,"
        "avg_frame_rate,r_frame_rate,duration_ts,duration,color_range,color_space,"
        "color_transfer,color_primaries:stream_tags=rotate,DURATION:"
        "stream_side_data=side_data_type,rotation,red_x,red_y,green_x,green_y,"
        "blue_x,blue_y,white_point_x,white_point_y,min_luminance,max_luminance,"
        "max_content,max_average:format=start_time,duration,format_name",
        "-of",
        "json=compact=1",
        "-protocol_whitelist",
        "file",
        path_utf8(media_path),
      };
    }

    void remove_temporary_probe_best_effort(const fs::path &path) noexcept {
      std::error_code ignored;
      fs::remove(path, ignored);
    }

    void remove_temporary_probe_checked(const fs::path &path) {
      std::error_code ec;
      const bool exists = fs::exists(path, ec);
      if (ec) {
        throw worker_error(
          "cannot inspect temporary probe path: " + path_utf8(path)
        );
      }
      if (exists && (!fs::remove(path, ec) || ec)) {
        throw worker_error(
          "cannot release temporary probe: " + path_utf8(path)
        );
      }
    }

    media_contract_t probe_media(
      const worker_spec_t &spec,
      const fs::path &media_path,
      const fs::path &json_path,
      const fs::path &log_path,
      const bool source_packet_durations
    ) {
      auto metadata_path = json_path;
      metadata_path += ".metadata";
      auto metadata_log = log_path;
      metadata_log.replace_filename(
        log_path.stem().string() + "-metadata" + log_path.extension().string()
      );
      try {
        remove_temporary_probe_checked(metadata_path);
        remove_temporary_probe_checked(json_path);
        const auto metadata = run_logged_json_bounded(
          build_ffprobe_metadata_command(spec, media_path),
          spec.sunshine_executable.parent_path(),
          metadata_log,
          max_video_metadata_probe_bytes,
          "FFprobe video metadata"
        );
        if (
          !metadata.is_object() ||
          !metadata.contains("streams") ||
          !metadata["streams"].is_array() ||
          metadata["streams"].size() != 1 ||
          !metadata["streams"][0].is_object()
        ) {
          throw worker_error(
            "FFprobe must return exactly one video stream descriptor"
          );
        }
        const auto stream = metadata["streams"][0];

        const auto deadline =
          std::chrono::steady_clock::now() + child_timeout;
        auto child = child_process_t::launch(
          build_ffprobe_command(spec, media_path),
          spec.sunshine_executable.parent_path(),
          log_path,
          std::nullopt,
          true
        );
        child_stdout_streambuf_t stdout_buffer(child, deadline);
        std::istream stdout_stream(&stdout_buffer);
        stdout_stream.exceptions(std::ios::badbit);
        auto result = parse_ffprobe_frames_incrementally(
          stream,
          stdout_stream,
          source_packet_durations,
          nullptr,
          "FFprobe stdout"
        );
        const auto now = std::chrono::steady_clock::now();
        const auto remaining = now < deadline ?
                                 std::chrono::duration_cast<
                                   std::chrono::milliseconds
                                 >(deadline - now) :
                                 std::chrono::milliseconds {0};
        const int code = child.wait(remaining);
        if (code != 0) {
          throw worker_error(
            "child process exited " + std::to_string(code) +
            "; see " + path_utf8(log_path)
          );
        }
        child.terminate();
        remove_temporary_probe_best_effort(json_path);
        return result;
      } catch (...) {
        // Raw FFprobe documents are transient validation inputs, not durable evidence. Keep
        // bounded normalized contracts/logs instead, including on malformed media.
        remove_temporary_probe_best_effort(metadata_path);
        remove_temporary_probe_best_effort(json_path);
        throw;
      }
    }

    struct packet_timing_t {
      std::optional<std::int64_t> pts;
      std::optional<std::int64_t> dts;
      std::optional<std::int64_t> duration;
    };
    static_assert(
      max_auxiliary_packets * sizeof(packet_timing_t) <=
        max_retained_packet_bytes,
      "the auxiliary packet vector exceeds its retained-memory contract"
    );

    struct stream_inventory_entry_t {
      std::int64_t index = -1;
      std::string type;
      std::string codec;
      rational_t time_base;
      nlohmann::json tags = nlohmann::json::object();
      nlohmann::json disposition = nlohmann::json::object();
      std::vector<packet_timing_t> packets;
    };

    struct chapter_inventory_entry_t {
      rational_t time_base;
      std::int64_t start = 0;
      std::int64_t end = 0;
      nlohmann::json tags = nlohmann::json::object();
    };

    struct stream_inventory_t {
      std::vector<stream_inventory_entry_t> streams;
      std::vector<chapter_inventory_entry_t> chapters;
      nlohmann::json format_tags = nlohmann::json::object();
    };

    struct streaming_packet_probe_stats_t {
      std::size_t peak_retained_packet_descriptors = 0;
      std::uintmax_t probe_bytes = 0;
      std::uint64_t packet_count = 0;
    };

    using packet_consumer_t = std::function<void(
      std::int64_t,
      packet_timing_t
    )>;

    std::uint64_t parse_ffprobe_packets_incrementally(
      std::istream &input,
      std::uint64_t max_packets,
      std::uintmax_t max_probe_bytes,
      const packet_consumer_t &consume,
      streaming_packet_probe_stats_t *stats,
      std::string_view description
    );

    streaming_packet_probe_stats_t run_ffprobe_packets_incrementally(
      const std::vector<std::string> &command,
      const fs::path &working_directory,
      const fs::path &log_path,
      std::uint64_t max_packets,
      std::uintmax_t max_probe_bytes,
      const packet_consumer_t &consume
    );

    nlohmann::json normalized_tags(
      const nlohmann::json &value
    ) {
      nlohmann::json result = nlohmann::json::object();
      if (!value.is_object()) {
        return result;
      }
      static const std::set<std::string> volatile_tags {
        "_statistics_tags",
        "_statistics_writing_app",
        "_statistics_writing_date_utc",
        "bps",
        "compatible_brands",
        "duration",
        "encoder",
        "major_brand",
        "minor_version",
        "number_of_bytes",
        "number_of_frames",
      };
      for (const auto &[key, item] : value.items()) {
        const auto normalized_key = lower(key);
        if (volatile_tags.contains(normalized_key)) {
          continue;
        }
        result[normalized_key] = item;
      }
      return result;
    }

    nlohmann::json normalized_disposition(
      const nlohmann::json &value
    ) {
      nlohmann::json result = nlohmann::json::object();
      if (!value.is_object()) {
        return result;
      }
      for (const auto &[key, item] : value.items()) {
        bool enabled = false;
        if (item.is_boolean()) {
          enabled = item.get<bool>();
        } else if (item.is_number_integer()) {
          enabled = item.get<std::int64_t>() != 0;
        }
        if (enabled) {
          result[lower(key)] = true;
        }
      }
      return result;
    }

    stream_inventory_t parse_stream_inventory(
      const nlohmann::json &value
    ) {
      if (!value.is_object() || !value.contains("streams") || !value["streams"].is_array()) {
        throw worker_error("FFprobe stream inventory is malformed");
      }
      if (value["streams"].size() > max_inventory_streams) {
        throw worker_error(
          "FFprobe stream inventory exceeds its 256-stream bound"
        );
      }
      stream_inventory_t result;
      std::map<std::int64_t, std::size_t> by_index;
      for (const auto &stream : value["streams"]) {
        if (!stream.is_object()) {
          throw worker_error("FFprobe inventory stream is malformed");
        }
        stream_inventory_entry_t entry;
        entry.index = stream.at("index").get<std::int64_t>();
        entry.type = stream.at("codec_type").get<std::string>();
        entry.codec = stream.value("codec_name", "");
        if (entry.index < 0 || entry.type.empty() || entry.codec.empty() || by_index.contains(entry.index)) {
          throw worker_error("FFprobe inventory stream identity is invalid");
        }
        if (const auto *time_base = object_member(stream, "time_base"); time_base && !unknown_metadata(*time_base)) {
          entry.time_base = parse_rational(*time_base, "stream time_base");
        }
        if (const auto *tags = object_member(stream, "tags")) {
          entry.tags = normalized_tags(*tags);
        }
        if (const auto *disposition = object_member(stream, "disposition")) {
          entry.disposition = normalized_disposition(*disposition);
        }
        by_index.emplace(entry.index, result.streams.size());
        result.streams.push_back(std::move(entry));
      }
      if (const auto *packets = object_member(value, "packets")) {
        if (!packets->is_array()) {
          throw worker_error("FFprobe packet inventory is malformed");
        }
        if (
          !can_retain_auxiliary_packets(
            0,
            static_cast<std::uint64_t>(packets->size())
          )
        ) {
          throw worker_error(
            "auxiliary packet inventory exceeds its two-million-packet bound"
          );
        }
        for (const auto &packet : *packets) {
          if (!packet.is_object()) {
            throw worker_error("FFprobe packet descriptor is malformed");
          }
          const auto stream_index =
            packet.at("stream_index").get<std::int64_t>();
          const auto found = by_index.find(stream_index);
          if (found == by_index.end()) {
            throw worker_error("FFprobe packet references an unknown stream");
          }
          result.streams[found->second].packets.push_back({
            .pts = optional_integer_json(packet, "pts"),
            .dts = optional_integer_json(packet, "dts"),
            .duration = optional_integer_json(packet, "duration"),
          });
        }
      }
      if (const auto *chapters = object_member(value, "chapters")) {
        if (!chapters->is_array()) {
          throw worker_error("FFprobe chapter inventory is malformed");
        }
        if (chapters->size() > max_inventory_chapters) {
          throw worker_error(
            "FFprobe chapter inventory exceeds its 65536-chapter bound"
          );
        }
        for (const auto &chapter : *chapters) {
          if (!chapter.is_object()) {
            throw worker_error("FFprobe chapter descriptor is malformed");
          }
          chapter_inventory_entry_t entry;
          entry.time_base =
            parse_rational(chapter.at("time_base"), "chapter time_base");
          const auto start = optional_integer_json(chapter, "start");
          const auto end = optional_integer_json(chapter, "end");
          if (!start || !end) {
            throw worker_error("FFprobe chapter timestamps are missing");
          }
          entry.start = *start;
          entry.end = *end;
          if (entry.end < entry.start) {
            throw worker_error("FFprobe chapter interval is invalid");
          }
          if (const auto *tags = object_member(chapter, "tags")) {
            entry.tags = normalized_tags(*tags);
          }
          result.chapters.push_back(std::move(entry));
        }
      }
      if (const auto *format = object_member(value, "format"); format && format->is_object()) {
        if (const auto *tags = object_member(*format, "tags")) {
          result.format_tags = normalized_tags(*tags);
        }
      }
      return result;
    }

    stream_inventory_t probe_stream_inventory(
      const worker_spec_t &spec,
      const fs::path &media,
      const fs::path &json_path,
      const fs::path &log_path
    ) {
      const std::vector<std::string> command {
        path_utf8(spec.ffprobe_executable),
        "-v",
        "error",
        "-show_streams",
        "-show_chapters",
        "-show_format",
        "-show_entries",
        "stream=index,codec_name,codec_type,time_base:"
        "stream_tags:stream_disposition:"
        "chapter=id,time_base,start,end:chapter_tags:format_tags",
        "-of",
        "json=compact=1",
        "-protocol_whitelist",
        "file",
        path_utf8(media),
      };
      std::vector<fs::path> transient_paths {json_path};
      try {
        remove_temporary_probe_checked(json_path);
        auto result = parse_stream_inventory(
          run_logged_json_bounded(
            command,
            spec.sunshine_executable.parent_path(),
            log_path,
            max_stream_inventory_probe_bytes,
            "FFprobe stream inventory"
          )
        );
        std::uint64_t total_packets = 0;
        std::uintmax_t total_packet_probe_bytes = 0;
        std::map<std::int64_t, std::size_t> by_index;
        for (std::size_t index = 0; index < result.streams.size(); ++index) {
          by_index.emplace(result.streams[index].index, index);
        }
        for (const auto &[type, selector] :
             std::array<std::pair<std::string_view, std::string_view>, 3> {{
               {"audio", "a"},
               {"subtitle", "s"},
               {"data", "d"},
             }}) {
          if (std::none_of(result.streams.begin(), result.streams.end(), [&](const stream_inventory_entry_t &stream) {
                return stream.type == type;
              })) {
            continue;
          }
          const auto packet_log =
            log_path.parent_path() /
            (json_path.stem().string() + "-packets-" +
             std::string(type) + ".log");
          if (total_packet_probe_bytes >= max_packet_probe_bytes) {
            throw worker_error(
              "FFprobe auxiliary packet documents exceed their shared " +
              std::to_string(max_packet_probe_bytes) + "-byte contract"
            );
          }
          const auto packet_stats = run_ffprobe_packets_incrementally(
            {
              path_utf8(spec.ffprobe_executable),
              "-v",
              "error",
              "-select_streams",
              std::string(selector),
              "-show_packets",
              "-show_entries",
              "packet=stream_index,pts,dts,duration",
              "-of",
              "json=compact=1",
              "-protocol_whitelist",
              "file",
              path_utf8(media),
            },
            spec.sunshine_executable.parent_path(),
            packet_log,
            max_auxiliary_packets - total_packets,
            max_packet_probe_bytes - total_packet_probe_bytes,
            [&](const std::int64_t stream_index, packet_timing_t timing) {
              const auto found = by_index.find(stream_index);
              if (
                found == by_index.end() ||
                result.streams[found->second].type != type
              ) {
                throw worker_error(
                  "FFprobe auxiliary packet identity is malformed"
                );
              }
              result.streams[found->second].packets.push_back(
                std::move(timing)
              );
            }
          );
          if (!can_consume_packet_probe_bytes(
                total_packet_probe_bytes,
                packet_stats.probe_bytes
              )) {
            throw worker_error(
              "FFprobe auxiliary packet documents exceed their shared " +
              std::to_string(max_packet_probe_bytes) + "-byte contract"
            );
          }
          total_packet_probe_bytes += packet_stats.probe_bytes;
          total_packets += packet_stats.packet_count;
        }
        for (const auto &path : transient_paths) {
          remove_temporary_probe_checked(path);
        }
        return result;
      } catch (...) {
        for (const auto &path : transient_paths) {
          remove_temporary_probe_best_effort(path);
        }
        throw;
      }
    }

    std::string output_container(const fs::path &path) {
      const auto extension = lower(path.extension().string());
      if (extension == ".mkv") {
        return "matroska";
      }
      if (extension == ".mp4") {
        return "mp4";
      }
      throw worker_error("offline output must use .mkv or .mp4");
    }

    void validate_output_time_base(
      const worker_spec_t &spec,
      const media_contract_t &media
    ) {
      if (!spec.staging_output) {
        return;
      }
      validate_avexpr_timeline_exactness(media);
      (void) output_container(*spec.staging_output);
      if (media.time_base.denominator > std::numeric_limits<std::int32_t>::max()) {
        throw worker_error(
          "source time-base denominator exceeds the MP4 encoder limit"
        );
      }
    }

    void require_tags_preserved(
      const nlohmann::json &source,
      const nlohmann::json &output,
      const std::string_view description
    ) {
      for (const auto &[key, value] : source.items()) {
        const auto found = output.find(key);
        if (found == output.end() || *found != value) {
          throw worker_error(
            std::string(description) + " metadata was not preserved: " + key
          );
        }
      }
    }

    std::vector<const stream_inventory_entry_t *> streams_of_type(
      const stream_inventory_t &inventory,
      const std::string_view type
    ) {
      std::vector<const stream_inventory_entry_t *> result;
      for (const auto &stream : inventory.streams) {
        if (stream.type == type) {
          result.push_back(&stream);
        }
      }
      return result;
    }

    void validate_stream_preflight(
      const worker_spec_t &spec,
      const stream_inventory_t &inventory
    ) {
      if (streams_of_type(inventory, "video").size() != 1) {
        throw worker_error(
          "offline conversion requires exactly one source video stream"
        );
      }
      static const std::set<std::string> known_types {
        "video",
        "audio",
        "subtitle",
        "data",
        "attachment",
      };
      for (const auto &stream : inventory.streams) {
        if (!known_types.contains(stream.type)) {
          throw worker_error(
            "unsupported source stream type would be dropped: " + stream.type
          );
        }
      }
      const auto container = output_container(*spec.staging_output);
      if (container == "mp4") {
        static const std::set<std::string> mp4_audio_codecs {
          "aac",
          "ac3",
          "alac",
          "eac3",
          "flac",
          "mp2",
          "mp3",
          "opus",
          "pcm_f32le",
          "pcm_s16le",
          "pcm_s24le",
          "pcm_s32le",
        };
        for (const auto *audio : streams_of_type(inventory, "audio")) {
          if (!mp4_audio_codecs.contains(audio->codec)) {
            throw worker_error(
              "MP4 cannot copy source audio codec '" + audio->codec +
              "' exactly; choose MKV or transcode the source audio first"
            );
          }
        }
        for (const auto *subtitle : streams_of_type(inventory, "subtitle")) {
          if (subtitle->codec != "mov_text") {
            throw worker_error(
              "MP4 cannot copy source subtitle codec '" + subtitle->codec +
              "' exactly; choose MKV or convert subtitles to mov_text"
            );
          }
        }
        if (!streams_of_type(inventory, "data").empty() || !streams_of_type(inventory, "attachment").empty()) {
          throw worker_error(
            "MP4 cannot preserve source data/attachment streams; choose MKV"
          );
        }
      } else if (!streams_of_type(inventory, "data").empty()) {
        throw worker_error(
          "FFmpeg Matroska cannot preserve this source data stream exactly"
        );
      }
    }

    nlohmann::json validate_stream_inventory(
      const stream_inventory_t &source,
      const stream_inventory_t &output,
      const std::uint64_t tolerance_actual_ticks
    ) {
      long double max_packet_pts_error = 0;
      long double max_packet_dts_error = 0;
      long double max_packet_duration_error = 0;
      long double max_stream_duration_error = 0;
      long double max_chapter_error = 0;
      std::uint64_t copied_packet_count = 0;
      if (tolerance_actual_ticks != 0) {
        for (const auto &stream : output.streams) {
          if (!stream.packets.empty()) {
            require_millisecond_or_finer(
              stream.time_base,
              "output auxiliary stream"
            );
          }
        }
        for (const auto &chapter : output.chapters) {
          require_millisecond_or_finer(
            chapter.time_base,
            "output chapter"
          );
        }
      }
      require_tags_preserved(
        source.format_tags,
        output.format_tags,
        "global"
      );
      for (const auto type : {
             std::string_view {"video"},
             std::string_view {"audio"},
             std::string_view {"subtitle"},
             std::string_view {"data"},
             std::string_view {"attachment"},
           }) {
        const auto wanted = streams_of_type(source, type);
        const auto actual = streams_of_type(output, type);
        if (wanted.size() != actual.size()) {
          throw worker_error(
            "output " + std::string(type) + " stream count differs from source"
          );
        }
        for (std::size_t index = 0; index < wanted.size(); ++index) {
          if (type != "video" && wanted[index]->codec != actual[index]->codec) {
            throw worker_error(
              "output " + std::string(type) +
              " codec differs from the copied source stream"
            );
          }
          require_tags_preserved(
            wanted[index]->tags,
            actual[index]->tags,
            std::string(type) + " stream"
          );
          const auto expected_disposition =
            type == "subtitle" ? nlohmann::json::object() : wanted[index]->disposition;
          if (expected_disposition != actual[index]->disposition) {
            throw worker_error(
              "output " + std::string(type) +
              (
                type == "subtitle" ?
                  " stream disposition was not cleared" :
                  " stream disposition differs from source"
              )
            );
          }
          if (type == "video") {
            continue;
          }
          if (wanted[index]->packets.size() != actual[index]->packets.size()) {
            throw worker_error(
              "output " + std::string(type) +
              " packet count differs from source"
            );
          }
          for (std::size_t packet_index = 0;
               packet_index < wanted[index]->packets.size();
               ++packet_index) {
            const auto &wanted_packet = wanted[index]->packets[packet_index];
            const auto &actual_packet = actual[index]->packets[packet_index];
            const auto compare = [&](const std::optional<std::int64_t> expected, const std::optional<std::int64_t> got, const std::string_view field, long double &max_error) {
              if (!expected) {
                return;
              }
              if (!got || !rational_ticks_within_actual_ticks(*expected, wanted[index]->time_base, *got, actual[index]->time_base, tolerance_actual_ticks)) {
                throw worker_error(
                  "output " + std::string(type) + " packet " +
                  std::string(field) + " differs from source"
                );
              }
              max_error = std::max(
                max_error,
                rational_error_in_actual_ticks(
                  *expected,
                  wanted[index]->time_base,
                  *got,
                  actual[index]->time_base
                )
              );
            };
            compare(
              wanted_packet.pts,
              actual_packet.pts,
              "PTS",
              max_packet_pts_error
            );
            compare(
              wanted_packet.dts,
              actual_packet.dts,
              "DTS",
              max_packet_dts_error
            );
            compare(
              wanted_packet.duration,
              actual_packet.duration,
              "duration",
              max_packet_duration_error
            );
            ++copied_packet_count;
          }
          const auto stream_span = [](const stream_inventory_entry_t &stream) {
            std::optional<std::int64_t> first;
            std::optional<std::int64_t> end;
            for (const auto &packet : stream.packets) {
              if (!packet.pts) {
                continue;
              }
              first = first ? std::min(*first, *packet.pts) : packet.pts;
              auto packet_end = *packet.pts;
              if (packet.duration && *packet.duration > 0) {
                if (packet_end > std::numeric_limits<std::int64_t>::max() - *packet.duration) {
                  throw worker_error(
                    "auxiliary packet end timestamp overflows"
                  );
                }
                packet_end += *packet.duration;
              }
              end = end ? std::max(*end, packet_end) :
                          std::optional<std::int64_t>(packet_end);
            }
            return std::pair {first, end};
          };
          const auto wanted_span = stream_span(*wanted[index]);
          const auto actual_span = stream_span(*actual[index]);
          if (wanted_span.first.has_value() != actual_span.first.has_value() || wanted_span.second.has_value() != actual_span.second.has_value()) {
            throw worker_error(
              "output auxiliary stream duration cannot be attested"
            );
          }
          if (wanted_span.first && wanted_span.second) {
            using boost::multiprecision::cpp_int;
            const cpp_int wanted_span_ticks =
              cpp_int(*wanted_span.second) - *wanted_span.first;
            const cpp_int actual_span_ticks =
              cpp_int(*actual_span.second) - *actual_span.first;
            if (wanted_span_ticks < 0 || actual_span_ticks < 0 || wanted_span_ticks > std::numeric_limits<std::int64_t>::max() || actual_span_ticks > std::numeric_limits<std::int64_t>::max()) {
              throw worker_error(
                "auxiliary stream duration is invalid"
              );
            }
            const auto wanted_duration =
              wanted_span_ticks.convert_to<std::int64_t>();
            const auto actual_duration =
              actual_span_ticks.convert_to<std::int64_t>();
            if (!rational_ticks_within_actual_ticks(
                  wanted_duration,
                  wanted[index]->time_base,
                  actual_duration,
                  actual[index]->time_base,
                  tolerance_actual_ticks
                )) {
              throw worker_error(
                "output auxiliary stream accumulated timeline drift"
              );
            }
            max_stream_duration_error = std::max(
              max_stream_duration_error,
              rational_error_in_actual_ticks(
                wanted_duration,
                wanted[index]->time_base,
                actual_duration,
                actual[index]->time_base
              )
            );
          }
        }
      }
      if (source.chapters.size() != output.chapters.size()) {
        throw worker_error("output chapter count differs from source");
      }
      for (std::size_t index = 0; index < source.chapters.size(); ++index) {
        const auto &wanted = source.chapters[index];
        const auto &actual = output.chapters[index];
        if (!rational_ticks_within_actual_ticks(wanted.start, wanted.time_base, actual.start, actual.time_base, tolerance_actual_ticks) || !rational_ticks_within_actual_ticks(wanted.end, wanted.time_base, actual.end, actual.time_base, tolerance_actual_ticks)) {
          throw worker_error("output chapter timing differs from source");
        }
        max_chapter_error = std::max({
          max_chapter_error,
          rational_error_in_actual_ticks(
            wanted.start,
            wanted.time_base,
            actual.start,
            actual.time_base
          ),
          rational_error_in_actual_ticks(
            wanted.end,
            wanted.time_base,
            actual.end,
            actual.time_base
          ),
        });
        require_tags_preserved(
          wanted.tags,
          actual.tags,
          "chapter"
        );
      }
      return {
        {"copied_packet_count", copied_packet_count},
        {"max_packet_pts_error_output_ticks",
         static_cast<double>(max_packet_pts_error)},
        {"max_packet_dts_error_output_ticks",
         static_cast<double>(max_packet_dts_error)},
        {"max_packet_duration_error_output_ticks",
         static_cast<double>(max_packet_duration_error)},
        {"max_stream_duration_error_output_ticks",
         static_cast<double>(max_stream_duration_error)},
        {"max_chapter_error_output_ticks",
         static_cast<double>(max_chapter_error)},
      };
    }

    std::vector<std::string> build_mux_command(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const stream_inventory_t &source_inventory,
      const fs::path &encoded_video
    ) {
      if (!spec.staging_output) {
        throw worker_error("conversion staging output is missing");
      }
      const auto container = output_container(*spec.staging_output);
      std::vector<std::string> command {
        path_utf8(spec.ffmpeg_executable),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-xerror",
        "-nostdin",
        // The worker atomically reserves and identity-pins this exact staging file
        // before launching any media tool. FFmpeg must truncate that owned placeholder.
        "-y",
        "-copyts",
        "-protocol_whitelist",
        "file",
        "-i",
        path_utf8(encoded_video),
        "-protocol_whitelist",
        "file",
        "-i",
        path_utf8(spec.input_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-map",
        "1:s?",
      };
      if (container == "matroska") {
        command.insert(command.end(), {
                                        "-map",
                                        "1:t?",
                                      });
      }
      command.insert(command.end(), {
                                      "-map_metadata",
                                      "-1",
                                      "-map_chapters",
                                      "1",
                                      "-c",
                                      "copy",
                                      "-avoid_negative_ts",
                                      "disabled",
                                    });
      const auto source_video = streams_of_type(source_inventory, "video");
      if (source_video.size() != 1) {
        throw worker_error("source video disposition contract is missing");
      }
      const auto append_metadata =
        [&](const std::string &specifier, const nlohmann::json &tags) {
          for (const auto &[name, value] : tags.items()) {
            if (!value.is_string()) {
              throw worker_error(
                "source metadata value is not textual: " + name
              );
            }
            command.insert(command.end(), {
                                            specifier,
                                            name + "=" + value.get<std::string>(),
                                          });
          }
        };
      const auto append_disposition =
        [&](const std::string &specifier, const nlohmann::json &disposition) {
          std::string enabled_dispositions;
          for (const auto &[name, enabled] : disposition.items()) {
            if (enabled.is_boolean() && enabled.get<bool>()) {
              if (!enabled_dispositions.empty()) {
                enabled_dispositions.push_back('+');
              }
              enabled_dispositions += name;
            }
          }
          command.insert(command.end(), {
                                          specifier,
                                          enabled_dispositions.empty() ? "0" : enabled_dispositions,
                                        });
        };
      const auto append_stream_contracts =
        [&](const std::string_view type, const std::string_view ffmpeg_type, const bool preserve_disposition = true) {
          const auto streams = streams_of_type(source_inventory, type);
          for (std::size_t index = 0; index < streams.size(); ++index) {
            const auto stream_index = std::to_string(index);
            append_metadata(
              "-metadata:s:" + std::string(ffmpeg_type) + ":" + stream_index,
              streams[index]->tags
            );
            if (preserve_disposition) {
              append_disposition(
                "-disposition:" + std::string(ffmpeg_type) + ":" + stream_index,
                streams[index]->disposition
              );
            }
          }
        };
      append_metadata("-metadata", source_inventory.format_tags);
      append_stream_contracts("video", "v");
      append_stream_contracts("audio", "a");
      const auto subtitle_streams = streams_of_type(source_inventory, "subtitle");
      append_stream_contracts("subtitle", "s", false);
      if (!subtitle_streams.empty()) {
        // A player cannot safely render one soft-subtitle plane over a packed SBS
        // raster. Preserve the streams and their metadata, but remove default/forced
        // disposition signals that can drive automatic selection. A player may still
        // select a stream through explicit language or user policy.
        command.insert(command.end(), {"-disposition:s", "0"});
      }
      if (container == "matroska") {
        append_stream_contracts("attachment", "t");
      }
      for (std::size_t index = 0; index < source_inventory.chapters.size(); ++index) {
        append_metadata(
          "-metadata:c:" + std::to_string(index),
          source_inventory.chapters[index].tags
        );
      }
      if (container == "mp4") {
        if (spec.codec == "hevc_nvenc") {
          command.insert(command.end(), {"-tag:v", "hvc1"});
        }
        command.insert(command.end(), {
                                        "-video_track_timescale",
                                        std::to_string(media.time_base.denominator),
                                        "-movie_timescale",
                                        std::to_string(media.time_base.denominator),
                                        "-movflags",
                                        "+faststart+write_colr",
                                        "-f",
                                        "mp4",
                                      });
      } else {
        command.insert(command.end(), {"-f", "matroska"});
      }
      command.push_back(path_utf8(*spec.staging_output));
      return command;
    }

    void mux_encoded_video(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const stream_inventory_t &source_inventory,
      const fs::path &encoded_video,
      const fs::path &log_path
    ) {
      std::error_code ec;
      if (!fs::is_regular_file(encoded_video, ec) || ec || fs::file_size(encoded_video, ec) == 0 || ec) {
        throw worker_error("conversion produced no compressed video");
      }
      run_logged(
        build_mux_command(spec, media, source_inventory, encoded_video),
        spec.sunshine_executable.parent_path(),
        log_path
      );
    }

    void validate_hdr_equivalence(
      const media_contract_t &source,
      const media_contract_t &output
    ) {
      if (source.color != output.color || source.color_range != output.color_range || source.color_space != output.color_space || source.color_transfer != output.color_transfer || source.color_primaries != output.color_primaries || source.mastering_display != output.mastering_display || source.content_light_level != output.content_light_level) {
        throw worker_error(
          "encoded HDR/static metadata differs from the source contract"
        );
      }
      if (source.static_side_data_types != output.static_side_data_types) {
        throw worker_error(
          "encoded HDR static side-data types differ from the source contract"
        );
      }
      if (!looks_high_bit_depth(output.pixel_format)) {
        throw worker_error("encoded HDR output is not a 10-bit profile");
      }
    }

    void validate_sdr_color_equivalence(
      const media_contract_t &source,
      const media_contract_t &output
    ) {
      if (source.color != media_color_e::sdr || output.color != media_color_e::sdr) {
        throw worker_error("SDR color validation received a non-SDR stream");
      }
      const auto differs = [](const std::string &wanted, const std::string &actual) {
        return !wanted.empty() && wanted != actual;
      };
      if (differs(source.color_range, output.color_range) || differs(source.color_space, output.color_space) || differs(source.color_transfer, output.color_transfer) || differs(source.color_primaries, output.color_primaries)) {
        throw worker_error("encoded SDR color tags differ from the source");
      }
    }

    nlohmann::json scene_audit_document(
      const std::vector<scene_plan_t> &scenes,
      const std::vector<boundary_audit_t> &boundaries,
      const std::string_view status,
      const std::uint64_t peak_cache_bytes,
      const std::uint64_t analysis_source_raster_bytes,
      const std::uint64_t peak_live_raster_bytes,
      const std::uint64_t peak_cache_plus_raster_bytes,
      const std::uint64_t hard_cap,
      const nlohmann::json &timeline_contract
    ) {
      nlohmann::json scene_values = nlohmann::json::array();
      for (const auto &scene : scenes) {
        scene_values.push_back(scene_json(scene));
      }
      nlohmann::json boundary_values = nlohmann::json::array();
      for (const auto &boundary : boundaries) {
        boundary_values.push_back(boundary_json(boundary));
      }
      return {
        {"schema", 2},
        {"version", "whole-clip-scene-audit-v2"},
        {"status", status},
        {"claims", {
                     {"ground_truth", false},
                     {"best_parameters", false},
                   }},
        {"policy", {
                     {"implementation", "native-offline-scene-planner"},
                     {"version", "scene-plan-v2"},
                     {"lookahead", true},
                     {"boundary_only", true},
                     {"python_dependency", false},
                   }},
        {"cache", {
                    {"peak_bytes", peak_cache_bytes},
                    {"analysis_source_raster_bytes", analysis_source_raster_bytes},
                    {"peak_live_raster_bytes", peak_live_raster_bytes},
                    {"peak_cache_plus_raster_bytes", peak_cache_plus_raster_bytes},
                    {"hard_cap_bytes", hard_cap},
                  }},
        {"timeline_contract", timeline_contract},
        {"scenes", std::move(scene_values)},
        {"boundary_revisions", std::move(boundary_values)},
      };
    }

    void account_serialized_record(
      const nlohmann::json &record,
      std::uintmax_t &payload_bytes,
      const std::uintmax_t document_limit,
      const std::string_view description
    ) {
      const auto serialized = record.dump(2);
      // Array members gain indentation when embedded in the final document. Eight
      // extra bytes per line exceeds the nesting delta of both result contracts.
      const auto bytes =
        static_cast<std::uintmax_t>(serialized.size()) +
        static_cast<std::uintmax_t>(
          std::count(serialized.begin(), serialized.end(), '\n')
        ) * 8ull +
        2ull;
      const auto payload_limit =
        document_limit - serialized_contract_fixed_reserve_bytes;
      if (
        bytes > payload_limit ||
        payload_bytes > payload_limit - bytes
      ) {
        throw worker_error(
          std::string {description} +
          " records exceed their bounded serialized contract"
        );
      }
      payload_bytes += bytes;
    }

    struct render_result_t {
      nlohmann::json contract;
      std::uint64_t peak_live_raster_bytes = 0;
      std::uint64_t peak_cache_plus_raster_bytes = 0;
    };

    fs::path replay_scene_log_path(const fs::path &work) {
      // Replays are strictly serial. Reuse one bounded diagnostic path so up to 1,920
      // successful scenes cannot retain 1,920 independent 8 MiB logs.
      return work / "logs" / "render-current-scene.log";
    }

    fs::path prepare_replay_scene_log(const fs::path &work) {
      const auto path = replay_scene_log_path(work);
      // bounded_child_log_t publishes only after the child exits. Clear the prior scene before
      // launch so a hard failure while the new child is still running cannot leave stale output
      // mislabeled as evidence for the current scene.
      remove_file_if_present_checked(path);
      return path;
    }

    render_result_t render_scene(
      const worker_spec_t &spec,
      const media_contract_t &media,
      const scene_plan_t &scene,
      streaming_decoder_t &render_decoder,
      whole_clip_encoder_t &encoder,
      const fs::path &work,
      const fs::path &cache,
      const std::uint32_t expected_sbs_width,
      const std::uint32_t expected_sbs_height,
      const std::uint64_t live_cache_bytes
    ) {
      const std::string output_extension =
        media.color == media_color_e::sdr ? "png" : "pfm";
      const auto scene_name =
        "scene_" + std::string(8 - std::min<std::size_t>(8, std::to_string(scene.scene_id).size()), '0') + std::to_string(scene.scene_id);
      const auto input = work / "render-input" / scene_name;
      const auto output = work / "render-output" / scene_name;
      const auto plan = work / "scene-plans" / (scene_name + ".json");
      std::error_code ec;
      fs::create_directories(input, ec);
      fs::create_directories(output, ec);
      fs::create_directories(plan.parent_path(), ec);
      if (ec) {
        throw worker_error("cannot create scene replay directories");
      }
      write_json_atomic(plan, scene_plan_json(scene));
      std::vector<std::string> command {
        path_utf8(spec.sunshine_executable),
        path_utf8(spec.sunshine_config),
        "--sbs-bench",
        "--frames",
        path_utf8(input),
        "--follow",
        "--follow-format",
        render_decoder.format(),
        "--follow-count",
        std::to_string(scene.frame_count),
        "--out",
        path_utf8(output),
        "--artifacts",
        "conversion",
        "--render-cache",
        path_utf8(cache),
        "--scene-plan",
        path_utf8(plan),
      };
      const auto replay_log = prepare_replay_scene_log(work);
      auto replay = child_process_t::launch(
        command,
        spec.sunshine_executable.parent_path(),
        replay_log
      );
      bool producer_done = false;
      std::uint64_t peak_live_raster_bytes = 0;
      std::uint64_t peak_cache_plus_raster_bytes = live_cache_bytes;
      try {
        for (std::uint64_t sequence = scene.start_sequence;
             sequence < scene.end_sequence_exclusive;
             ++sequence) {
          const auto source = render_decoder.publish_next(input, sequence);
          const auto progress = read_progress(
            output / "follow_progress.json",
            replay,
            sequence - scene.start_sequence + 1,
            "scene replay"
          );
          if (progress.value("first_sequence", 0ull) != scene.start_sequence || progress.value("last_completed_sequence", 0ull) != sequence || progress.value("artifact_mode", "") != "conversion") {
            throw worker_error("scene replay global sequence ACK mismatch");
          }
          const auto sbs = output /
                           ("sbs_" + frame_id(sequence) + "." + output_extension);
          if (!fs::is_regular_file(sbs, ec) || ec || fs::file_size(sbs, ec) == 0 || ec) {
            throw worker_error("scene replay ACK lacks its atomic SBS frame");
          }
          const auto source_bytes = fs::file_size(source, ec);
          if (ec) {
            throw worker_error("cannot measure the replay source raster");
          }
          const auto sbs_bytes = fs::file_size(sbs, ec);
          if (ec || source_bytes > std::numeric_limits<std::uint64_t>::max() || sbs_bytes > std::numeric_limits<std::uint64_t>::max()) {
            throw worker_error("live replay raster size is unsupported");
          }
          const auto source_size =
            static_cast<std::uint64_t>(source_bytes);
          const auto sbs_size =
            static_cast<std::uint64_t>(sbs_bytes);
          if (source_size > spec.scene_cache_hard_cap_bytes || sbs_size > spec.scene_cache_hard_cap_bytes - source_size || live_cache_bytes > spec.scene_cache_hard_cap_bytes - source_size - sbs_size) {
            throw worker_error(
              "scene cache plus live replay rasters exceeded the hard cap"
            );
          }
          peak_live_raster_bytes = std::max(
            peak_live_raster_bytes,
            source_size + sbs_size
          );
          peak_cache_plus_raster_bytes = std::max(
            peak_cache_plus_raster_bytes,
            live_cache_bytes + source_size + sbs_size
          );
          encoder.publish(sequence, sbs);
          remove_file_checked(sbs);
          remove_file_checked(source);
        }
        publish_producer_done(input, scene.frame_count);
        producer_done = true;
        const int code = replay.wait(
          std::chrono::duration_cast<std::chrono::milliseconds>(child_timeout)
        );
        if (code != 0) {
          throw worker_error(
            "scene replay exited " + std::to_string(code)
          );
        }
        replay.terminate();
      } catch (const std::exception &exception) {
        if (!producer_done) {
          publish_producer_failed(input, exception.what());
        }
        replay.terminate();
        throw;
      }

      const auto contract = read_json(output / "whole_clip_contract.json");
      if (contract.value("schema", 0) != 1 || contract.value("artifact_mode", "") != "conversion" || contract.value("source_frame_count", 0ull) != scene.frame_count || contract.value("source_first_sequence", 0ull) != scene.start_sequence || contract.value("inference_mode", "") != "scene-cache-replay" || contract.value("depth_inference_enabled", true) || contract.value("scheduled_depth_update_count", 1ull) != 0 || contract.value("tensorrt_enqueue_count", 1ull) != 0) {
        throw worker_error(
          "scene replay did not attest a zero-inference exact cache replay"
        );
      }
      if (
        !contract.contains("source_scope") ||
        !offline_full_frame_source_scope_is_valid(contract["source_scope"])
      ) {
        throw worker_error(
          "scene replay did not attest selected-input full-frame isolation"
        );
      }
      const auto &adaptive_state = contract.at("adaptive_state");
      if (
        !adaptive_state.is_object() ||
        adaptive_state.value("transport", "") != "none" ||
        adaptive_state.value("retained_history", true) ||
        adaptive_state.value("frame_count", 1ull) != 0ull ||
        adaptive_state.size() != 3u ||
        contract.contains("cut_state")
      ) {
        throw worker_error(
          "scene replay falsely attributed adaptive-state evidence"
        );
      }
      const auto &sbs = contract.at("sbs");
      if (!sbs.value("enabled", false) || sbs.value("frame_count", 0ull) != scene.frame_count || sbs.value("width", 0u) != expected_sbs_width || sbs.value("height", 0u) != expected_sbs_height || sbs.value("file_pattern", "") != "sbs_<frame-id>." + output_extension) {
        throw worker_error("scene replay SBS raster contract mismatch");
      }

      // The contract has been copied into bounded in-memory evidence and every SBS frame has
      // already been consumed by the persistent encoder. Remove the fixed replay transport
      // files now so even a clip with many short scenes cannot accumulate one trace snapshot
      // or progress file per scene while the job is still running.
      for (const auto &path : {
             output / "follow_progress.json",
             output / "sbs_perf.json",
             output / "whole_clip_contract.json",
             input / ".producer-done.json",
             plan,
           }) {
        remove_file_if_present_checked(path);
      }
      remove_empty_directory_checked(output);
      remove_empty_directory_checked(input);

      // The persistent encoder has consumed every SBS raster before the scene cache is
      // released. It retains only compressed encoder state, never a scene of rasters.
      release_cache_scene(cache, scene);
      return {
        .contract = contract,
        .peak_live_raster_bytes = peak_live_raster_bytes,
        .peak_cache_plus_raster_bytes =
          peak_cache_plus_raster_bytes,
      };
    }
  }  // namespace

  std::vector<std::string> build_codec_arguments(
    const worker_spec_t &spec,
    const media_contract_t &media,
    const bool hdr,
    const std::uint32_t encoded_width,
    const std::uint32_t encoded_height
  ) {
    return codec_arguments_impl(
      spec,
      media,
      hdr,
      encoded_width,
      encoded_height
    );
  }

#ifdef SUNSHINE_TESTS
  bool native_stdout_pipe_error_is_eof_for_test(
    const std::uint32_t error
  ) noexcept {
#ifdef _WIN32
    return native_stdout_pipe_error_is_eof(static_cast<DWORD>(error));
#else
    (void) error;
    return false;
#endif
  }

  bool adaptive_trace_flags_valid_for_test(
    const float cut_flags,
    const std::uint32_t analysis_flags
  ) {
    return adaptive_trace_flags_valid(cut_flags, analysis_flags);
  }

  std::uint64_t retained_timing_frame_limit_for_test() {
    return max_retained_timing_frames;
  }

  bool can_retain_timing_frame_for_test(
    const std::uint64_t retained_frames
  ) {
    return can_retain_another_timing_frame(retained_frames);
  }

  std::uintmax_t stream_inventory_probe_byte_limit_for_test() {
    return max_stream_inventory_probe_bytes;
  }

  std::uintmax_t packet_probe_byte_limit_for_test() {
    return max_packet_probe_bytes;
  }

  std::uint64_t auxiliary_packet_limit_for_test() {
    return max_auxiliary_packets;
  }

  std::uint64_t retained_packet_payload_limit_for_test() {
    return max_retained_packet_bytes;
  }

  std::size_t child_process_log_byte_limit_for_test() {
    return max_child_log_bytes;
  }

  std::string bound_child_process_log_for_test(
    const std::string_view bytes
  ) {
    bounded_log_accumulator_t accumulator;
    constexpr std::size_t chunk_bytes = 3137;
    for (std::size_t offset = 0; offset < bytes.size();) {
      const auto size = std::min<std::size_t>(
        chunk_bytes,
        bytes.size() - offset
      );
      accumulator.append(bytes.data() + offset, size);
      offset += size;
    }
    const auto result = accumulator.render();
    return std::string(result.begin(), result.end());
  }

  fs::path replay_scene_log_path_for_test(
    const fs::path &work,
    const std::uint64_t scene_id
  ) {
    (void) scene_id;
    return replay_scene_log_path(work);
  }

  fs::path prepare_replay_scene_log_for_test(
    const fs::path &work,
    const std::uint64_t scene_id
  ) {
    (void) scene_id;
    return prepare_replay_scene_log(work);
  }

  bool can_retain_auxiliary_packets_for_test(
    const std::uint64_t retained_packets,
    const std::uint64_t additional_packets
  ) {
    return can_retain_auxiliary_packets(
      retained_packets,
      additional_packets
    );
  }

  bool can_consume_packet_probe_bytes_for_test(
    const std::uintmax_t consumed_bytes,
    const std::uintmax_t additional_bytes
  ) {
    return can_consume_packet_probe_bytes(
      consumed_bytes,
      additional_bytes
    );
  }

  std::vector<std::string> build_mux_command_for_test(
    const worker_spec_t &spec,
    const media_contract_t &media,
    const nlohmann::json &source_inventory,
    const fs::path &encoded_video
  ) {
    return build_mux_command(
      spec,
      media,
      parse_stream_inventory(source_inventory),
      encoded_video
    );
  }

  streaming_probe_test_result_t
  parse_and_remove_ffprobe_frames_for_test(
    const nlohmann::json &stream,
    const fs::path &frames_path,
    const bool require_source_packet_durations
  ) {
    streaming_probe_stats_t stats;
    try {
      auto media = parse_ffprobe_frames_incrementally(
        stream,
        frames_path,
        require_source_packet_durations,
        &stats
      );
      const auto compact_contract_bytes =
        media_contract_json(media).dump().size();
      remove_file_checked(frames_path);
      return {
        .media = std::move(media),
        .peak_retained_frame_descriptors =
          stats.peak_retained_frame_descriptors,
        .probe_bytes = stats.probe_bytes,
        .compact_contract_bytes = compact_contract_bytes,
      };
    } catch (...) {
      remove_temporary_probe_best_effort(frames_path);
      throw;
    }
  }

  streaming_packet_probe_test_result_t
  parse_and_remove_ffprobe_packets_for_test(
    const fs::path &packets_path,
    const std::uint64_t max_packets
  ) {
    streaming_packet_probe_stats_t stats;
    try {
      std::error_code ec;
      if (!fs::is_regular_file(packets_path, ec) || ec) {
        throw worker_error(
          "FFprobe packet document is missing: " + path_utf8(packets_path)
        );
      }
      std::ifstream input(packets_path, std::ios::binary);
      if (!input) {
        throw worker_error(
          "cannot open FFprobe packet document: " + path_utf8(packets_path)
        );
      }
      const auto count = parse_ffprobe_packets_incrementally(
        input,
        max_packets,
        max_packet_probe_bytes,
        [](std::int64_t, packet_timing_t) {},
        &stats,
        path_utf8(packets_path)
      );
      input.close();
      remove_file_checked(packets_path);
      return {
        .packet_count = count,
        .peak_retained_packet_descriptors =
          stats.peak_retained_packet_descriptors,
        .probe_bytes = stats.probe_bytes,
      };
    } catch (...) {
      remove_temporary_probe_best_effort(packets_path);
      throw;
    }
  }

  void validate_windows_command_line_capacity_for_test(
    const std::vector<std::string> &arguments
  ) {
#ifdef _WIN32
    validate_windows_command_line_capacity(arguments, "test command");
#else
    (void) arguments;
#endif
  }

#ifdef _WIN32
  std::string secure_frame_bridge_token_for_test() {
    return secure_frame_bridge_token();
  }

  bool frame_bridge_survives_unauthenticated_disconnect_for_test() {
    scene_frame_server_t server(1, 2, "png");
    const auto connect_client = [&]() {
      SOCKET client = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (client == INVALID_SOCKET) {
        return client;
      }
      sockaddr_in address {};
      address.sin_family = AF_INET;
      address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      address.sin_port = htons(server.port_for_test());
      if (connect(
            client,
            reinterpret_cast<const sockaddr *>(&address),
            sizeof(address)
          ) != 0) {
        closesocket(client);
        return INVALID_SOCKET;
      }
      return client;
    };

    SOCKET rogue = connect_client();
    if (rogue == INVALID_SOCKET) {
      server.abort("test connection failed");
      return false;
    }
    shutdown(rogue, SD_BOTH);
    closesocket(rogue);

    SOCKET probe = connect_client();
    if (probe == INVALID_SOCKET) {
      server.abort("test probe connection failed");
      return false;
    }
    DWORD timeout_ms = 5000;
    setsockopt(
      probe,
      SOL_SOCKET,
      SO_RCVTIMEO,
      reinterpret_cast<const char *>(&timeout_ms),
      sizeof(timeout_ms)
    );
    constexpr std::string_view request {
      "GET /unauthenticated HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
    };
    const bool request_sent =
      send(
        probe,
        request.data(),
        static_cast<int>(request.size()),
        0
      ) == static_cast<int>(request.size());
    std::array<char, 256> response {};
    const int received =
      request_sent ?
        recv(probe, response.data(), static_cast<int>(response.size()), 0) :
        -1;
    shutdown(probe, SD_BOTH);
    closesocket(probe);
    server.abort("test completed");
    return received > 0 &&
           std::string_view(response.data(), static_cast<std::size_t>(received))
             .starts_with("HTTP/1.1 404");
  }
#endif
#endif

  double rational_t::seconds(const std::int64_t ticks) const {
    if (numerator <= 0 || denominator <= 0) {
      throw worker_error("invalid media time base");
    }
    return static_cast<double>(ticks) *
           static_cast<double>(numerator) /
           static_cast<double>(denominator);
  }

  std::int64_t media_contract_t::first_pts() const {
    if (frames.empty()) {
      throw worker_error("media contract has no frames");
    }
    return frames.front().pts;
  }

  std::int64_t media_contract_t::end_pts_exclusive() const {
    if (frames.empty()) {
      throw worker_error("media contract has no frames");
    }
    const auto &last = frames.back();
    if (last.duration > 0 && last.pts <= std::numeric_limits<std::int64_t>::max() - last.duration) {
      return last.pts + last.duration;
    }
    throw worker_error("media contract end timestamp overflows");
  }

  double media_contract_t::duration_seconds() const {
    return time_base.seconds(
      nonnegative_tick_difference(
        end_pts_exclusive(),
        first_pts(),
        "media presentation duration"
      )
    );
  }

  bool media_contract_t::variable_frame_rate() const {
    if (frames.size() < 3) {
      return false;
    }
    const auto duration = frames.front().duration;
    return std::any_of(
      frames.begin() + 1,
      frames.end() - 1,
      [duration](const frame_timing_t &frame) {
        return frame.duration != duration;
      }
    );
  }

  worker_spec_t parse_worker_spec(const nlohmann::json &value) {
    if (!value.is_object() || value.value("schema", 0) != 1 || value.value("python_dependency", true)) {
      throw worker_error("worker specification is not native schema 1");
    }
    worker_spec_t spec;
    spec.job_id = required_string(value, "job_id");
    if (spec.job_id.size() > 128) {
      throw worker_error("worker job identity is too long");
    }
    spec.operation = required_string(value, "operation");
    if (spec.operation != "evaluate" && spec.operation != "convert") {
      throw worker_error("worker operation must be evaluate or convert");
    }
    spec.input_path = required_absolute_path(value, "input_path");
    spec.job_directory = required_absolute_path(value, "job_directory");
    spec.result_directory = required_absolute_path(value, "result_directory");
    spec.progress_path = required_absolute_path(value, "progress_path");
    spec.result_path = required_absolute_path(value, "result_path");
    if (!path_is_within(spec.result_directory, spec.job_directory) || !path_is_within(spec.progress_path, spec.job_directory) || !path_is_within(spec.result_path, spec.job_directory)) {
      throw worker_error(
        "worker state/result paths must remain inside the job directory"
      );
    }
    const auto *staging = object_member(value, "staging_output");
    if (staging && !staging->is_null()) {
      if (!staging->is_string()) {
        throw worker_error("staging_output must be an absolute path or null");
      }
      const auto path = path_from_utf8(staging->get<std::string>());
      if (!path.is_absolute()) {
        throw worker_error("staging_output must be absolute");
      }
      spec.staging_output = path.lexically_normal();
    }
    if ((spec.operation == "convert") != spec.staging_output.has_value()) {
      throw worker_error(
        "conversion requires staging_output and evaluation forbids it"
      );
    }
    const auto &sunshine = value.at("sunshine");
    spec.sunshine_executable =
      required_absolute_path(sunshine, "executable");
    spec.sunshine_config = required_absolute_path(sunshine, "config");
    spec.ffmpeg_executable =
      required_absolute_path(value.at("ffmpeg"), "path");
    spec.ffprobe_executable =
      required_absolute_path(value.at("ffprobe"), "path");
    spec.codec = required_string(value, "codec");
    if (spec.codec != "hevc_nvenc" && spec.codec != "av1_nvenc") {
      throw worker_error("worker codec must be hevc_nvenc or av1_nvenc");
    }
    const auto &cache = value.at("scene_cache");
    spec.scene_cache_hard_cap_bytes =
      cache.at("hard_cap_bytes").get<std::uint64_t>();
    if (spec.scene_cache_hard_cap_bytes < 16ull * 1024ull * 1024ull) {
      throw worker_error("scene-cache hard cap is below 16 MiB");
    }
    const auto policy = required_string(cache, "budget_policy");
    if (policy != "fail" && policy != "split") {
      throw worker_error("scene-cache budget policy is invalid");
    }
    spec.allow_administrative_split = policy == "split";
    const auto &planner = value.at("planner");
    if (
      required_string(planner, "implementation") != "native-offline-scene-planner" ||
      required_string(planner, "scene_plan_contract") != "scene-plan-v2"
    ) {
      throw worker_error("worker scene planner contract is unsupported");
    }
    return spec;
  }

  worker_spec_t read_authenticated_worker_spec(
    const fs::path &path,
    const std::string_view expected_sha256
  ) {
    if (!valid_sha256_hex(expected_sha256)) {
      throw worker_error(
        "worker specification SHA-256 must be 64 lowercase hexadecimal characters"
      );
    }
    const auto bytes = read_bounded_bytes(path, max_spec_bytes);
    const auto actual_sha256 = sha256_hex(bytes);
    if (!constant_time_equal(actual_sha256, expected_sha256)) {
      throw worker_error(
        "worker specification authentication failed; exact bytes changed after admission"
      );
    }
    try {
      auto spec = parse_worker_spec(parse_json_without_duplicate_keys(bytes));
      spec.authenticated_spec_sha256 = actual_sha256;
      return spec;
    } catch (const worker_error &) {
      throw;
    } catch (const std::exception &exception) {
      throw worker_error(
        std::string {"cannot parse authenticated worker specification: "} +
        exception.what()
      );
    }
  }

  namespace {
    class bounded_probe_streambuf_t: public std::streambuf {
    public:
      bounded_probe_streambuf_t(
        std::streambuf &source,
        const std::size_t max_descriptor_bytes,
        const std::optional<std::uintmax_t> max_total_bytes = std::nullopt,
        std::string descriptor_name = "frame"
      ):
          source_(source),
          max_descriptor_bytes_(max_descriptor_bytes),
          max_total_bytes_(max_total_bytes),
          descriptor_name_(std::move(descriptor_name)) {
        if (
          max_descriptor_bytes_ == 0 ||
          (max_total_bytes_ && *max_total_bytes_ == 0)
        ) {
          throw worker_error("FFprobe probe bound is invalid");
        }
        setg(buffer_.data(), buffer_.data(), buffer_.data());
      }

      void begin_descriptor() {
        if (descriptor_start_) {
          throw worker_error(
            "FFprobe " + descriptor_name_ + " descriptors overlap"
          );
        }
        descriptor_start_ = consumed_bytes();
      }

      void end_descriptor() {
        descriptor_start_.reset();
      }

      void begin_frame() {
        begin_descriptor();
      }

      void end_frame() {
        end_descriptor();
      }

      [[nodiscard]] std::uintmax_t fetched_bytes() const {
        return fetched_bytes_;
      }

    protected:
      int_type underflow() override {
        if (gptr() < egptr()) {
          return traits_type::to_int_type(*gptr());
        }
        auto request = buffer_.size();
        if (max_total_bytes_) {
          if (fetched_bytes_ >= *max_total_bytes_) {
            // Distinguish an exactly-bound valid document from a document with a hidden
            // suffix. sgetc() does not consume the suffix, so diagnostics and cleanup retain
            // deterministic behavior while the resident buffer stays fixed-size.
            if (
              source_.sgetc() !=
              std::streambuf::traits_type::eof()
            ) {
              throw worker_error(
                "FFprobe document exceeds its " +
                std::to_string(*max_total_bytes_) +
                "-byte contract"
              );
            }
            return traits_type::eof();
          }
          request = std::min<std::size_t>(
            request,
            static_cast<std::size_t>(
              *max_total_bytes_ - fetched_bytes_
            )
          );
        }
        if (descriptor_start_) {
          const auto consumed = consumed_bytes();
          if (
            consumed < *descriptor_start_ ||
            consumed - *descriptor_start_ >= max_descriptor_bytes_
          ) {
            throw worker_error(
              "FFprobe " + descriptor_name_ +
              " descriptor exceeds its bounded contract"
            );
          }
          request = std::min<std::size_t>(
            request,
            max_descriptor_bytes_ -
              (consumed - *descriptor_start_)
          );
        }
        const auto received = source_.sgetn(
          buffer_.data(),
          static_cast<std::streamsize>(request)
        );
        if (received <= 0) {
          return traits_type::eof();
        }
        fetched_bytes_ += static_cast<std::uint64_t>(received);
        setg(
          buffer_.data(),
          buffer_.data(),
          buffer_.data() + received
        );
        return traits_type::to_int_type(*gptr());
      }

    private:
      std::uint64_t consumed_bytes() const {
        return fetched_bytes_ -
               static_cast<std::uint64_t>(egptr() - gptr());
      }

      std::streambuf &source_;
      std::size_t max_descriptor_bytes_ = 0;
      std::optional<std::uintmax_t> max_total_bytes_;
      std::string descriptor_name_;
      std::array<char, 4ull * 1024ull> buffer_ {};
      std::uintmax_t fetched_bytes_ = 0;
      std::optional<std::uintmax_t> descriptor_start_;
    };

    std::uint64_t parse_ffprobe_packets_incrementally(
      std::istream &input,
      const std::uint64_t max_packets,
      const std::uintmax_t max_probe_bytes,
      const packet_consumer_t &consume,
      streaming_packet_probe_stats_t *stats,
      const std::string_view description
    ) {
      constexpr std::size_t max_packet_descriptor_bytes =
        64ull * 1024ull;
      if (
        max_packets > max_auxiliary_packets ||
        max_probe_bytes == 0 ||
        max_probe_bytes > max_packet_probe_bytes ||
        !consume
      ) {
        throw worker_error(
          "auxiliary packet parser has an invalid bounded contract"
        );
      }

      bounded_probe_streambuf_t bounded_buffer(
        *input.rdbuf(),
        max_packet_descriptor_bytes,
        max_probe_bytes,
        "auxiliary packet"
      );
      std::istream bounded_input(&bounded_buffer);
      bounded_input.exceptions(std::ios::badbit);

      bool expect_packets_array = false;
      bool saw_top_level_object = false;
      bool finished_top_level_object = false;
      bool in_packets_array = false;
      bool saw_packets_array = false;
      int packets_array_depth = -1;
      int packet_object_depth = -1;
      std::size_t live_packet_descriptors = 0;
      std::uint64_t packet_count = 0;
      const auto callback = [&](
                              const int depth,
                              const nlohmann::json::parse_event_t event,
                              nlohmann::json &parsed) {
        switch (event) {
          case nlohmann::json::parse_event_t::key:
            if (
              saw_top_level_object &&
              !finished_top_level_object &&
              !in_packets_array &&
              depth == 1
            ) {
              if (
                !parsed.is_string() ||
                parsed.get_ref<const std::string &>() != "packets"
              ) {
                throw worker_error(
                  "FFprobe packet document has an unexpected top-level member"
                );
              }
              expect_packets_array = true;
            } else {
              expect_packets_array = false;
            }
            break;
          case nlohmann::json::parse_event_t::array_start:
            if (expect_packets_array) {
              if (saw_packets_array) {
                throw worker_error(
                  "FFprobe packet document contains multiple packet arrays"
                );
              }
              saw_packets_array = true;
              in_packets_array = true;
              packets_array_depth = depth;
              expect_packets_array = false;
            } else if (in_packets_array) {
              throw worker_error(
                "FFprobe auxiliary packet descriptor is malformed"
              );
            }
            break;
          case nlohmann::json::parse_event_t::object_start:
            if (depth == 0) {
              if (saw_top_level_object) {
                throw worker_error(
                  "FFprobe packet document has multiple top-level objects"
                );
              }
              saw_top_level_object = true;
            } else if (expect_packets_array) {
              throw worker_error(
                "FFprobe top-level packets member is not an array"
              );
            } else if (
              in_packets_array &&
              packet_object_depth < 0 &&
              depth == packets_array_depth + 1
            ) {
              packet_object_depth = depth;
              bounded_buffer.begin_descriptor();
              ++live_packet_descriptors;
              if (stats) {
                stats->peak_retained_packet_descriptors = std::max(
                  stats->peak_retained_packet_descriptors,
                  live_packet_descriptors
                );
              }
            } else if (in_packets_array) {
              throw worker_error(
                "FFprobe auxiliary packet descriptor is malformed"
              );
            }
            break;
          case nlohmann::json::parse_event_t::object_end:
            if (
              in_packets_array &&
              depth == packet_object_depth
            ) {
              if (
                parsed.dump().size() > max_packet_descriptor_bytes
              ) {
                throw worker_error(
                  "FFprobe auxiliary packet descriptor exceeds its bounded contract"
                );
              }
              if (packet_count >= max_packets) {
                throw worker_error(
                  "auxiliary packet inventory exceeds its two-million-packet bound"
                );
              }
              const auto stream_index =
                optional_integer_json(parsed, "stream_index");
              if (!stream_index || *stream_index < 0) {
                throw worker_error(
                  "FFprobe auxiliary packet identity is malformed"
                );
              }
              consume(
                *stream_index,
                {
                  .pts = optional_integer_json(parsed, "pts"),
                  .dts = optional_integer_json(parsed, "dts"),
                  .duration = optional_integer_json(parsed, "duration"),
                }
              );
              ++packet_count;
              bounded_buffer.end_descriptor();
              packet_object_depth = -1;
              --live_packet_descriptors;
              // Returning false discards this object from the parent array before the next
              // descriptor is parsed. Only the compact packet_timing_t supplied to the
              // consumer remains resident.
              return false;
            }
            if (depth == 0) {
              finished_top_level_object = true;
            }
            break;
          case nlohmann::json::parse_event_t::array_end:
            if (
              in_packets_array &&
              depth == packets_array_depth
            ) {
              in_packets_array = false;
              packets_array_depth = -1;
            }
            break;
          case nlohmann::json::parse_event_t::value:
            if (expect_packets_array) {
              throw worker_error(
                "FFprobe top-level packets member is not an array"
              );
            }
            if (in_packets_array && packet_object_depth < 0) {
              throw worker_error(
                "FFprobe auxiliary packet descriptor is malformed"
              );
            }
            break;
          default:
            break;
        }
        return true;
      };

      try {
        const auto discarded_document = nlohmann::json::parse(
          bounded_input,
          callback,
          true,
          true
        );
        (void) discarded_document;
      } catch (const worker_error &) {
        throw;
      } catch (const std::exception &exception) {
        throw worker_error(
          "cannot incrementally parse " + std::string(description) +
          ": " + exception.what()
        );
      }
      if (
        !saw_top_level_object ||
        !finished_top_level_object ||
        !saw_packets_array ||
        in_packets_array ||
        packet_object_depth >= 0 ||
        live_packet_descriptors != 0
      ) {
        throw worker_error("FFprobe packet document is malformed");
      }
      if (stats) {
        stats->probe_bytes = bounded_buffer.fetched_bytes();
        stats->packet_count = packet_count;
      }
      return packet_count;
    }

    streaming_packet_probe_stats_t run_ffprobe_packets_incrementally(
      const std::vector<std::string> &command,
      const fs::path &working_directory,
      const fs::path &log_path,
      const std::uint64_t max_packets,
      const std::uintmax_t max_probe_bytes,
      const packet_consumer_t &consume
    ) {
      const auto deadline =
        std::chrono::steady_clock::now() + child_timeout;
      auto child = child_process_t::launch(
        command,
        working_directory,
        log_path,
        std::nullopt,
        true
      );
      child_stdout_streambuf_t stdout_buffer(child, deadline);
      std::istream stdout_stream(&stdout_buffer);
      stdout_stream.exceptions(std::ios::badbit);
      streaming_packet_probe_stats_t stats;
      (void) parse_ffprobe_packets_incrementally(
        stdout_stream,
        max_packets,
        max_probe_bytes,
        consume,
        &stats,
        "FFprobe auxiliary packet stdout"
      );
      const auto now = std::chrono::steady_clock::now();
      const auto remaining = now < deadline ?
                               std::chrono::duration_cast<
                                 std::chrono::milliseconds
                               >(deadline - now) :
                               std::chrono::milliseconds {0};
      const int code = child.wait(remaining);
      if (code != 0) {
        throw worker_error(
          "child process exited " + std::to_string(code) +
          "; see " + path_utf8(log_path)
        );
      }
      child.terminate();
      return stats;
    }

    class ffprobe_contract_builder_t {
    public:
      ffprobe_contract_builder_t(
        nlohmann::json stream,
        const bool require_source_packet_durations
      ):
          stream_(std::move(stream)),
          require_source_packet_durations_(require_source_packet_durations) {
        if (!stream_.is_object()) {
          throw worker_error("FFprobe video stream is malformed");
        }
        reject_rotation(stream_);
        const auto stream_text = [&](const std::string_view name) {
          const auto *field = object_member(stream_, name);
          if (!field || unknown_metadata(*field)) {
            return std::string {};
          }
          if (!field->is_string()) {
            throw worker_error(
              "FFprobe " + std::string(name) + " is not textual"
            );
          }
          return field->get<std::string>();
        };
        const auto codec_identity =
          lower(
            stream_text("codec_name") + " " + stream_text("profile") + " " +
            stream_text("codec_tag_string")
          );
        if (
          codec_identity.find("dolby vision") != std::string::npos ||
          codec_identity.find("dovi") != std::string::npos ||
          codec_identity.find("dvhe") != std::string::npos ||
          codec_identity.find("dvh1") != std::string::npos
        ) {
          throw worker_error("Dolby Vision video is unsupported");
        }
        const auto sample_aspect_ratio =
          lower(stream_text("sample_aspect_ratio"));
        if (
          !sample_aspect_ratio.empty() &&
          sample_aspect_ratio != "1:1" &&
          sample_aspect_ratio != "0:1"
        ) {
          throw worker_error(
            "non-square-pixel video is unsupported; normalize sample aspect ratio first"
          );
        }
        const auto field_order = lower(stream_text("field_order"));
        if (!field_order.empty() && field_order != "progressive") {
          throw worker_error(
            "interlaced video is unsupported; deinterlace the source first"
          );
        }

        result_.width = stream_.at("width").get<std::uint32_t>();
        result_.height = stream_.at("height").get<std::uint32_t>();
        if (
          result_.width == 0 || result_.height == 0 ||
          result_.width > 16384 || result_.height > 16384
        ) {
          throw worker_error("FFprobe stream dimensions are unsupported");
        }
        result_.codec_name = stream_.value("codec_name", "");
        result_.time_base =
          parse_rational(stream_.at("time_base"), "time_base");
        inspect_side_data(stream_);
        merge_canonical_side_data(
          stream_,
          "Mastering display metadata",
          mastering_display_
        );
        merge_canonical_side_data(
          stream_,
          "Content light level metadata",
          content_light_level_
        );
      }

      void add_frame(const nlohmann::json &frame) {
        if (!frame.is_object()) {
          throw worker_error("FFprobe frame descriptor is malformed");
        }
        if (frame_count_ >= max_sequence) {
          throw worker_error(
            "source exceeds the 10-digit native sequence contract"
          );
        }
        if (const auto *interlaced = object_member(frame, "interlaced_frame")) {
          bool value = false;
          if (interlaced->is_boolean()) {
            value = interlaced->get<bool>();
          } else if (interlaced->is_number_integer()) {
            value = interlaced->get<std::int64_t>() != 0;
          } else {
            throw worker_error("FFprobe interlaced_frame is malformed");
          }
          if (value) {
            throw worker_error(
              "interlaced video is unsupported; deinterlace the source first"
            );
          }
        }
        if (
          const auto *aspect = object_member(frame, "sample_aspect_ratio");
          aspect && !unknown_metadata(*aspect)
        ) {
          if (!aspect->is_string()) {
            throw worker_error(
              "FFprobe frame sample_aspect_ratio is malformed"
            );
          }
          const auto value = lower(aspect->get<std::string>());
          if (value != "1:1" && value != "0:1") {
            throw worker_error(
              "decoded video sample aspect ratio changed or is non-square"
            );
          }
        }
        reject_rotation(frame);
        if (
          (
            frame.contains("width") &&
            frame.at("width").get<std::uint32_t>() != result_.width
          ) ||
          (
            frame.contains("height") &&
            frame.at("height").get<std::uint32_t>() != result_.height
          )
        ) {
          throw worker_error(
            "decoded video dimensions changed during the clip"
          );
        }

        if (!color_initialized_) {
          result_.pixel_format = color_property(stream_, frame, "pix_fmt");
          if (has_alpha_channel(result_.pixel_format)) {
            throw worker_error(
              "alpha-bearing video is unsupported because H.265/AV1 output "
              "cannot preserve transparency"
            );
          }
          result_.color_range =
            color_property(stream_, frame, "color_range");
          result_.color_space =
            color_property(stream_, frame, "color_space");
          result_.color_transfer =
            color_property(stream_, frame, "color_transfer");
          result_.color_primaries =
            color_property(stream_, frame, "color_primaries");
          classify_color();
          color_initialized_ = true;
        }
        const std::array<std::pair<std::string_view, const std::string *>, 5>
          stable_color_fields {{
            {"pix_fmt", &result_.pixel_format},
            {"color_range", &result_.color_range},
            {"color_space", &result_.color_space},
            {"color_transfer", &result_.color_transfer},
            {"color_primaries", &result_.color_primaries},
          }};
        for (const auto &[name, expected] : stable_color_fields) {
          const auto actual = color_property(stream_, frame, name);
          if (actual != *expected) {
            throw worker_error(
              "video color/pixel format changed during the clip: " +
              std::string(name)
            );
          }
        }

        inspect_side_data(frame);
        merge_canonical_side_data(
          frame,
          "Mastering display metadata",
          mastering_display_
        );
        merge_canonical_side_data(
          frame,
          "Content light level metadata",
          content_light_level_
        );

        const auto pts = optional_integer_json(frame, "pts").value_or(
          optional_integer_json(frame, "best_effort_timestamp").value_or(
            std::numeric_limits<std::int64_t>::min()
          )
        );
        if (pts == std::numeric_limits<std::int64_t>::min()) {
          throw worker_error(
            "FFprobe frame has no presentation timestamp"
          );
        }
        const auto reported_duration =
          optional_integer_json(frame, "duration");
        if (reported_duration && *reported_duration <= 0) {
          throw worker_error("FFprobe frame duration is not positive");
        }

        if (pending_) {
          if (pts <= pending_->pts) {
            throw worker_error(
              "FFprobe frame PTS are not strictly increasing in presentation order"
            );
          }
          const auto presentation_delta = positive_tick_difference(
            pts,
            pending_->pts,
            "FFprobe frame duration"
          );
          if (
            require_source_packet_durations_ &&
            pending_->reported_duration &&
            *pending_->reported_duration != presentation_delta
          ) {
            throw worker_error(
              "source contains a presentation gap/overlap which cannot be "
              "proven through image-sequence replay"
            );
          }
          const auto duration =
            !require_source_packet_durations_ &&
                pending_->reported_duration ?
              *pending_->reported_duration :
              presentation_delta;
          append_timing(pending_->pts, duration);
        } else {
          first_pts_ = pts;
        }
        pending_ = pending_timing_t {
          .pts = pts,
          .reported_duration = reported_duration,
        };
        ++frame_count_;
      }

      media_contract_t finish() {
        if (!pending_ || frame_count_ == 0 || !color_initialized_) {
          throw worker_error(
            "FFprobe must return one video stream and its complete frame list"
          );
        }
        auto duration = pending_->reported_duration;
        if (!duration && !require_source_packet_durations_) {
          if (
            const auto total =
              stream_duration_ticks(stream_, result_.time_base)
          ) {
            const auto preceding =
              result_.frames.empty() ?
                0 :
                positive_tick_difference(
                  pending_->pts,
                  first_pts_,
                  "FFprobe stream duration prefix"
                );
            if (*total > preceding) {
              duration = *total - preceding;
            }
          }
        }
        if (!duration || *duration <= 0) {
          throw worker_error(
            "FFprobe did not expose an exact positive final-frame duration"
          );
        }
        append_timing(pending_->pts, *duration);
        result_.frames.shrink_to_fit();

        result_.static_side_data_types.assign(
          side_types_.begin(),
          side_types_.end()
        );
        result_.dropped_nonsemantic_side_data_types.assign(
          dropped_side_types_.begin(),
          dropped_side_types_.end()
        );
        result_.mastering_display =
          mastering_display_.value_or(nlohmann::json(nullptr));
        result_.content_light_level =
          content_light_level_.value_or(nlohmann::json(nullptr));
        if (
          result_.color == media_color_e::sdr &&
          !result_.static_side_data_types.empty()
        ) {
          throw worker_error(
            "SDR video side data cannot be preserved through raster replay"
          );
        }
        positive_tick_difference(
          result_.end_pts_exclusive(),
          result_.first_pts(),
          "FFprobe total presentation span"
        );
        return std::move(result_);
      }

    private:
      struct pending_timing_t {
        std::int64_t pts = 0;
        std::optional<std::int64_t> reported_duration;
      };

      void inspect_side_data(const nlohmann::json &owner) {
        const auto *list = object_member(owner, "side_data_list");
        if (!list || !list->is_array()) {
          return;
        }
        for (const auto &item : *list) {
          const auto type = side_data_type(item);
          if (type.empty()) {
            continue;
          }
          if (dynamic_hdr_side_data(type)) {
            throw worker_error(
              "Dolby Vision/dynamic HDR metadata is unsupported: " + type
            );
          }
          if (
            !supported_static_video_side_data(type) &&
            !encoder_generated_side_data(type)
          ) {
            throw worker_error(
              "unsupported video side data cannot be preserved exactly: " +
              type
            );
          }
          if (supported_static_video_side_data(type)) {
            side_types_.insert(type);
          } else if (require_source_packet_durations_) {
            dropped_side_types_.insert(type);
          }
        }
      }

      void classify_color() {
        const auto transfer = lower(result_.color_transfer);
        const bool high_depth =
          looks_high_bit_depth(result_.pixel_format);
        if (transfer == "smpte2084" || transfer == "arib-std-b67") {
          if (!high_depth) {
            throw worker_error("PQ/HLG requires a high-bit-depth source");
          }
          if (
            result_.color_range.empty() ||
            result_.color_space.empty() ||
            result_.color_primaries.empty()
          ) {
            throw worker_error(
              "HDR requires explicit range, matrix, primaries, and transfer metadata"
            );
          }
          const auto primaries = lower(result_.color_primaries);
          const auto matrix = lower(result_.color_space);
          if (
            primaries != "bt2020" ||
            (matrix != "bt2020nc" && matrix != "bt2020c")
          ) {
            throw worker_error(
              "only explicitly tagged BT.2020 PQ/HLG video is supported"
            );
          }
          result_.color = transfer == "smpte2084" ?
                            media_color_e::hdr_pq :
                            media_color_e::hdr_hlg;
        } else {
          if (high_depth) {
            throw worker_error(
              "unknown/high-bit-depth non-PQ/HLG video is unsupported"
            );
          }
          result_.color = media_color_e::sdr;
        }
      }

      void append_timing(
        const std::int64_t pts,
        const std::int64_t duration
      ) {
        if (duration <= 0) {
          throw worker_error("FFprobe frame timing is outside its contract");
        }
        if (result_.frames.size() >= max_sequence) {
          throw worker_error(
            "FFprobe frame timing exceeds the 10-digit sequence contract"
          );
        }
        if (!can_retain_another_timing_frame(result_.frames.size())) {
          throw worker_error(
            "FFprobe frame timing exceeds the 128 MiB retained-memory contract"
          );
        }
        result_.frames.push_back({
          result_.frames.size() + 1,
          pts,
          duration,
        });
      }

      nlohmann::json stream_;
      bool require_source_packet_durations_ = true;
      media_contract_t result_;
      std::optional<pending_timing_t> pending_;
      std::int64_t first_pts_ = 0;
      std::uint64_t frame_count_ = 0;
      bool color_initialized_ = false;
      std::set<std::string> side_types_;
      std::set<std::string> dropped_side_types_;
      std::optional<nlohmann::json> mastering_display_;
      std::optional<nlohmann::json> content_light_level_;
    };

    media_contract_t parse_ffprobe_frames_incrementally(
      const nlohmann::json &stream,
      std::istream &input,
      const bool require_source_packet_durations,
      streaming_probe_stats_t *stats,
      const std::string_view description
    ) {
      constexpr std::size_t max_frame_descriptor_bytes =
        256ull * 1024ull;
      ffprobe_contract_builder_t builder(
        stream,
        require_source_packet_durations
      );
      bounded_probe_streambuf_t bounded_buffer(
        *input.rdbuf(),
        max_frame_descriptor_bytes
      );
      std::istream bounded_input(&bounded_buffer);
      bounded_input.exceptions(std::ios::badbit);
      bool expect_frames_array = false;
      bool saw_top_level_object = false;
      bool finished_top_level_object = false;
      bool in_frames_array = false;
      bool saw_frames_array = false;
      int frames_array_depth = -1;
      int frame_object_depth = -1;
      std::size_t live_frame_descriptors = 0;
      const auto callback = [&](
                              const int depth,
                              const nlohmann::json::parse_event_t event,
                              nlohmann::json &parsed) {
        switch (event) {
          case nlohmann::json::parse_event_t::key:
            if (
              saw_top_level_object &&
              !finished_top_level_object &&
              !in_frames_array &&
              depth == 1
            ) {
              if (
                !parsed.is_string() ||
                parsed.get_ref<const std::string &>() != "frames"
              ) {
                throw worker_error(
                  "FFprobe frame document has an unexpected top-level member"
                );
              }
              expect_frames_array = true;
            } else {
              expect_frames_array = false;
            }
            break;
          case nlohmann::json::parse_event_t::array_start:
            if (expect_frames_array) {
              if (saw_frames_array) {
                throw worker_error(
                  "FFprobe frame document contains multiple frame arrays"
                );
              }
              saw_frames_array = true;
              in_frames_array = true;
              frames_array_depth = depth;
              expect_frames_array = false;
            } else if (
              in_frames_array &&
              frame_object_depth < 0 &&
              depth == frames_array_depth + 1
            ) {
              throw worker_error(
                "FFprobe frame descriptor is malformed"
              );
            }
            break;
          case nlohmann::json::parse_event_t::object_start:
            if (depth == 0) {
              if (saw_top_level_object) {
                throw worker_error(
                  "FFprobe frame document has multiple top-level objects"
                );
              }
              saw_top_level_object = true;
            } else if (expect_frames_array) {
              throw worker_error(
                "FFprobe top-level frames member is not an array"
              );
            } else if (
              in_frames_array &&
              frame_object_depth < 0 &&
              depth == frames_array_depth + 1
            ) {
              frame_object_depth = depth;
              bounded_buffer.begin_frame();
              ++live_frame_descriptors;
              if (stats) {
                stats->peak_retained_frame_descriptors = std::max(
                  stats->peak_retained_frame_descriptors,
                  live_frame_descriptors
                );
              }
            }
            break;
          case nlohmann::json::parse_event_t::object_end:
            if (
              in_frames_array &&
              depth == frame_object_depth
            ) {
              if (
                parsed.dump().size() > max_frame_descriptor_bytes
              ) {
                throw worker_error(
                  "FFprobe frame descriptor exceeds its bounded contract"
                );
              }
              builder.add_frame(parsed);
              bounded_buffer.end_frame();
              frame_object_depth = -1;
              --live_frame_descriptors;
              // Discard the completed descriptor from the parent array. Nlohmann retains only
              // the current object while parsing; the normalized timing vector is the sole
              // per-frame resident contract.
              return false;
            }
            if (depth == 0) {
              finished_top_level_object = true;
            }
            break;
          case nlohmann::json::parse_event_t::array_end:
            if (
              in_frames_array &&
              depth == frames_array_depth
            ) {
              in_frames_array = false;
              frames_array_depth = -1;
            }
            break;
          case nlohmann::json::parse_event_t::value:
            if (expect_frames_array) {
              throw worker_error(
                "FFprobe top-level frames member is not an array"
              );
            }
            if (in_frames_array && frame_object_depth < 0) {
              throw worker_error(
                "FFprobe frame descriptor is malformed"
              );
            }
            break;
          default:
            break;
        }
        return true;
      };

      try {
        const auto discarded_document = nlohmann::json::parse(
          bounded_input,
          callback,
          true,
          true
        );
        (void) discarded_document;
      } catch (const worker_error &) {
        throw;
      } catch (const std::exception &exception) {
        throw worker_error(
          "cannot incrementally parse " + std::string(description) +
          ": " + exception.what()
        );
      }
      if (
        !saw_top_level_object ||
        !finished_top_level_object ||
        !saw_frames_array ||
        in_frames_array ||
        frame_object_depth >= 0 ||
        live_frame_descriptors != 0
      ) {
        throw worker_error("FFprobe frame document is malformed");
      }
      return builder.finish();
    }

#ifdef SUNSHINE_TESTS
    media_contract_t parse_ffprobe_frames_incrementally(
      const nlohmann::json &stream,
      const fs::path &frames_path,
      const bool require_source_packet_durations,
      streaming_probe_stats_t *stats
    ) {
      constexpr std::uintmax_t max_frame_probe_bytes =
        64ull * 1024ull * 1024ull;
      std::error_code ec;
      if (!fs::is_regular_file(frames_path, ec) || ec) {
        throw worker_error(
          "FFprobe frame document is missing: " + path_utf8(frames_path)
        );
      }
      const auto bytes = fs::file_size(frames_path, ec);
      if (ec || bytes == 0 || bytes > max_frame_probe_bytes) {
        throw worker_error(
          "FFprobe frame document exceeds its bounded storage contract"
        );
      }
      if (stats) {
        stats->probe_bytes = bytes;
      }
      std::ifstream input(frames_path, std::ios::binary);
      if (!input) {
        throw worker_error(
          "cannot open FFprobe frame document: " + path_utf8(frames_path)
        );
      }
      return parse_ffprobe_frames_incrementally(
        stream,
        input,
        require_source_packet_durations,
        stats,
        path_utf8(frames_path)
      );
    }
#endif
  }  // namespace

  media_contract_t parse_ffprobe_contract(
    const nlohmann::json &value,
    const bool require_source_packet_durations
  ) {
    if (
      !value.is_object() ||
      !value.contains("streams") ||
      !value["streams"].is_array() ||
      value["streams"].size() != 1 ||
      !value.contains("frames") ||
      !value["frames"].is_array() ||
      value["frames"].empty()
    ) {
      throw worker_error(
        "FFprobe must return one video stream and its complete frame list"
      );
    }
    ffprobe_contract_builder_t builder(
      value["streams"][0],
      require_source_packet_durations
    );
    for (const auto &frame : value["frames"]) {
      builder.add_frame(frame);
    }
    return builder.finish();
  }

  bool offline_full_frame_source_scope_is_valid(
    const nlohmann::json &value
  ) noexcept {
    try {
      return
        value.is_object() &&
        value.size() == 4u &&
        value.value("frame_source", std::string {}) == whole_clip_frame_source &&
        value.value("analysis_region", std::string {}) == whole_clip_analysis_region &&
        !value.value("active_window_dependency", true) &&
        !value.value("window_region_roi", true);
    } catch (const nlohmann::json::exception &) {
      return false;
    }
  }

  std::uint64_t analysis_open_cache_limit(
    const std::uint64_t hard_cap_bytes,
    const std::uint64_t source_raster_bytes,
    const std::uint64_t depth_state_pair_bytes
  ) {
    if (hard_cap_bytes == 0 || source_raster_bytes == 0 || depth_state_pair_bytes == 0 || source_raster_bytes >= hard_cap_bytes) {
      throw worker_error("analysis storage budget has an invalid byte contract");
    }
    const auto after_source = hard_cap_bytes - source_raster_bytes;
    if (depth_state_pair_bytes > after_source) {
      throw worker_error(
        "analysis hard cap cannot hold the live source raster and depth/state pair"
      );
    }
    const auto open_cache_limit = after_source - depth_state_pair_bytes;
    if (open_cache_limit < depth_state_pair_bytes) {
      throw worker_error(
        "analysis hard cap cannot reserve the next exact depth/state pair"
      );
    }
    return open_cache_limit;
  }

  void validate_avexpr_timeline_exactness(const media_contract_t &media) {
    using boost::multiprecision::cpp_int;
    static const cpp_int max_exact_avexpr_integer {
      "9007199254740991"
    };
    if (media.frames.empty() || media.frames.size() > max_sequence) {
      throw worker_error("cannot generate AVExpr for an invalid frame sequence");
    }
    for (const auto &frame : media.frames) {
      const cpp_int pts = frame.pts;
      const cpp_int duration = frame.duration;
      if (frame.duration <= 0 || boost::multiprecision::abs(pts) > max_exact_avexpr_integer || duration > max_exact_avexpr_integer) {
        throw worker_error(
          "source timestamps exceed AVExpr's exact integer range"
        );
      }
    }
    for (const auto &run : pts_runs(media)) {
      if (run.end <= run.begin || run.step <= 0) {
        throw worker_error("source timing run is invalid");
      }
      const cpp_int index_span = run.end - run.begin - 1;
      const cpp_int product = index_span * run.step;
      if (product > max_exact_avexpr_integer) {
        throw worker_error(
          "source timing run exceeds AVExpr's exact multiplication range"
        );
      }
      const cpp_int last_pts = cpp_int(run.first_pts) + product;
      if (boost::multiprecision::abs(last_pts) > max_exact_avexpr_integer) {
        throw worker_error(
          "source timing run exceeds AVExpr's exact addition range"
        );
      }
    }
  }

  void validate_timeline_equivalence(
    const media_contract_t &source,
    const media_contract_t &actual
  ) {
    validate_timeline_contract(source, actual, 1);
  }

  std::string select_av1_level(
    const media_contract_t &media,
    const std::uint32_t encoded_width,
    const std::uint32_t encoded_height
  ) {
    struct level_limit_t {
      const char *name;
      std::uint64_t max_picture_size;
      std::uint32_t max_width;
      std::uint32_t max_height;
      std::uint64_t max_display_rate;
      std::uint32_t max_header_rate;
    };

    // AV1 Bitstream & Decoding Process Specification, Annex A.  Reserved level
    // indices (including 7.x) deliberately do not appear here.
    static constexpr std::array levels {
      level_limit_t {"2.0", 147456, 2048, 1152, 4423680, 150},
      level_limit_t {"2.1", 278784, 2816, 1584, 8363520, 150},
      level_limit_t {"3.0", 665856, 4352, 2448, 19975680, 150},
      level_limit_t {"3.1", 1065024, 5504, 3096, 31950720, 150},
      level_limit_t {"4.0", 2359296, 6144, 3456, 70778880, 300},
      level_limit_t {"4.1", 2359296, 6144, 3456, 141557760, 300},
      level_limit_t {"5.0", 8912896, 8192, 4352, 267386880, 300},
      level_limit_t {"5.1", 8912896, 8192, 4352, 534773760, 300},
      level_limit_t {"5.2", 8912896, 8192, 4352, 1069547520, 300},
      level_limit_t {"5.3", 8912896, 8192, 4352, 1069547520, 300},
      level_limit_t {"6.0", 35651584, 16384, 8704, 1069547520, 300},
      level_limit_t {"6.1", 35651584, 16384, 8704, 2139095040, 300},
      level_limit_t {"6.2", 35651584, 16384, 8704, 4278190080, 300},
      level_limit_t {"6.3", 35651584, 16384, 8704, 4278190080, 300},
    };

    if (
      encoded_width == 0 || encoded_height == 0 || media.frames.empty() ||
      media.time_base.numerator <= 0 || media.time_base.denominator <= 0
    ) {
      throw worker_error("cannot select an AV1 level from an invalid media contract");
    }
    const auto shortest =
      std::min_element(
        media.frames.begin(),
        media.frames.end(),
        [](const frame_timing_t &left, const frame_timing_t &right) {
          return left.duration < right.duration;
        }
      )->duration;
    if (shortest <= 0) {
      throw worker_error("cannot select an AV1 level for a non-positive frame interval");
    }

    using boost::multiprecision::cpp_int;
    // NVENC validates AV1 levels against its coded superblock raster, not only
    // the visible frame dimensions. AV1 superblocks are at least 64x64, so a
    // visible height such as 180 is coded as 192 lines. Selecting against the
    // visible raster can therefore choose a boundary level that NVENC rejects.
    constexpr std::uint64_t av1_superblock_size = 64;
    const auto coded_width =
      (static_cast<std::uint64_t>(encoded_width) + av1_superblock_size - 1) /
      av1_superblock_size * av1_superblock_size;
    const auto coded_height =
      (static_cast<std::uint64_t>(encoded_height) + av1_superblock_size - 1) /
      av1_superblock_size * av1_superblock_size;
    const cpp_int picture_size = cpp_int(coded_width) * coded_height;
    // display_rate = picture_size * time_base.denominator /
    //                (shortest_duration * time_base.numerator).
    // Compare as exact integers so nanosecond and large-container time bases cannot
    // silently round a boundary down into the wrong level.
    const cpp_int display_rate_numerator =
      picture_size * media.time_base.denominator;
    const cpp_int interval_denominator =
      cpp_int(shortest) * media.time_base.numerator;

    for (const auto &level : levels) {
      if (
        picture_size > level.max_picture_size ||
        coded_width > level.max_width ||
        coded_height > level.max_height ||
        display_rate_numerator >
          cpp_int(level.max_display_rate) * interval_denominator ||
        media.time_base.denominator >
          cpp_int(level.max_header_rate) * interval_denominator
      ) {
        continue;
      }
      return level.name;
    }
    throw worker_error(
      "packed SBS coded raster/frame rate exceeds the defined AV1 level 6.3 limits"
    );
  }

  std::vector<std::string> build_ffprobe_command(
    const worker_spec_t &spec,
    const fs::path &media_path
  ) {
    return {
      path_utf8(spec.ffprobe_executable),
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_frames",
      "-show_entries",
      "frame=pts,best_effort_timestamp,duration,width,height,pix_fmt,"
      "sample_aspect_ratio,interlaced_frame,top_field_first,"
      "color_range,color_space,color_transfer,color_primaries:"
      "frame_side_data=side_data_type,rotation,red_x,red_y,green_x,green_y,"
      "blue_x,blue_y,white_point_x,white_point_y,min_luminance,max_luminance,"
      "max_content,max_average",
      "-of",
      "json=compact=1",
      "-protocol_whitelist",
      "file",
      path_utf8(media_path),
    };
  }

  std::vector<std::string> build_decoder_command(
    const worker_spec_t &spec,
    const media_contract_t &media
  ) {
    std::vector<std::string> command {
      path_utf8(spec.ffmpeg_executable),
      "-hide_banner",
      "-loglevel",
      "warning",
      "-xerror",
      "-nostdin",
      "-noautorotate",
      "-protocol_whitelist",
      "file",
      "-i",
      path_utf8(spec.input_path),
      "-map",
      "0:v:0",
      "-an",
      "-sn",
      "-dn",
      "-fps_mode",
      "passthrough",
    };
    if (media.color == media_color_e::sdr) {
      command.insert(command.end(), {
                                      "-pix_fmt",
                                      "bgra",
                                      "-f",
                                      "rawvideo",
                                      "pipe:1",
                                    });
    } else {
      command.insert(command.end(), {
                                      "-vf",
                                      hdr_decode_filter(media),
                                      "-c:v",
                                      "pfm",
                                      "-f",
                                      "image2pipe",
                                      "pipe:1",
                                    });
    }
    return command;
  }

  int run(const int argc, char **argv) {
    std::optional<worker_spec_t> parsed_spec;
    std::optional<staging_output_claim_t> staging_claim;
    try {
      if (
        argc != 2 || !argv ||
        !argv[0] || !*argv[0] ||
        !argv[1] || !*argv[1]
      ) {
        throw worker_error(
          "--offline-sbs-worker requires a worker-spec path and its SHA-256"
        );
      }
      const auto spec_path = fs::absolute(path_from_utf8(argv[0])).lexically_normal();
      parsed_spec = read_authenticated_worker_spec(spec_path, argv[1]);
      const auto &spec = *parsed_spec;
      std::error_code ec;
      if (!fs::is_regular_file(spec.input_path, ec) || ec || fs::file_size(spec.input_path, ec) == 0 || ec) {
        throw worker_error("offline input is missing or empty");
      }
      for (const auto &tool : {
             spec.sunshine_executable,
             spec.sunshine_config,
             spec.ffmpeg_executable,
             spec.ffprobe_executable,
           }) {
        if (!fs::is_regular_file(tool, ec) || ec) {
          throw worker_error("trusted worker executable/config is missing");
        }
      }
      if (spec.staging_output) {
        staging_claim.emplace(*spec.staging_output);
      }
      fs::create_directories(spec.result_directory, ec);
      if (ec) {
        throw worker_error("cannot create worker result directory");
      }
      const auto work = spec.job_directory / "native-work";
      if (fs::exists(work, ec) && !fs::is_empty(work, ec)) {
        throw worker_error("native worker directory must be new or empty");
      }
      fs::create_directories(work / "logs", ec);
      if (ec) {
        throw worker_error("cannot create native worker directory");
      }

      publish_progress(spec, "probe", 0, nullptr, 0);
      const auto probe_json = work / "source-probe.json";
      const auto media = probe_media(
        spec,
        spec.input_path,
        probe_json,
        work / "logs" / "source-probe.log",
        true
      );
      validate_output_time_base(spec, media);
      const auto observation_timeline = work / "source-observation-timeline.sbsotl";
      write_observation_timeline(observation_timeline, media);
      std::optional<stream_inventory_t> source_inventory;
      if (spec.operation == "convert") {
        source_inventory = probe_stream_inventory(
          spec,
          spec.input_path,
          work / "source-stream-inventory.json",
          work / "logs" / "source-stream-inventory.log"
        );
        validate_stream_preflight(spec, *source_inventory);
#ifdef _WIN32
        validate_windows_command_line_capacity(
          build_mux_command(
            spec,
            media,
            *source_inventory,
            work / "encoded-video.mp4"
          ),
          "final mux command; reduce source metadata"
        );
#endif
      }
      write_json_atomic(
        spec.result_directory / "source-contract.json",
        media_contract_json(media)
      );

      // Refuse to start TensorRT unless the exact cache/replay harness is present.
      const auto capabilities_path = spec.result_directory / "native-capabilities.json";
      run_logged(
        {
          path_utf8(spec.sunshine_executable),
          path_utf8(spec.sunshine_config),
          "--sbs-bench",
          "--capabilities",
          path_utf8(capabilities_path),
        },
        spec.sunshine_executable.parent_path(),
        work / "logs" / "native-capabilities.log"
      );
      const auto capabilities = read_json(capabilities_path);
      const auto &native = capabilities.at("native_whole_clip");
      if (capabilities.value("schema", 0) != 1 || native.value("follow_protocol_schema", 0) != 1 || native.value("adaptive_state_schema", 0u) != sbs_adaptive_state::schema_version || native.value("adaptive_state_contract_tag", 0u) != sbs_adaptive_state::cut_contract_tag || native.value("adaptive_state_contract_canonical_sha256", "") != sbs_adaptive_state::contract_canonical_sha256 || native.value("scene_cache_contract_schema", 0u) != scene_cache_contract_schema || native.value("renderer", "") != "depth-coordinate-v2-live-signed-parallax" || !native.value("render_cache_follow", false) || !native.value("render_skips_tensorrt", false) || !native.value("atomic_sbs_publication", false)) {
        throw worker_error("native SBS harness lacks the required replay contract");
      }
      if (
        !native.contains("source_scope") ||
        !offline_full_frame_source_scope_is_valid(native["source_scope"])
      ) {
        throw worker_error(
          "native SBS harness is not isolated to selected-input full-frame processing"
        );
      }
      const auto &scene_plan_capability = native.at("scene_plan");
      if (scene_plan_capability.value("schema", 0) != 2 ||
          scene_plan_capability.value("version", "") != "scene-plan-v2" ||
          !scene_plan_capability.value("one_scene_per_replay", false) ||
          !scene_plan_capability.value("boundary_only", false)) {
        throw worker_error("native SBS harness lacks boundary-only scene planning");
      }
      nlohmann::json expected_analysis_flag_bits = nlohmann::json::object();
      for (const auto &flag : sbs_adaptive_state::analysis_flag_bits) {
        expected_analysis_flag_bits[std::string {flag.name}] = flag.bit;
      }
      if (
        !native.contains("adaptive_analysis_flag_bits") ||
        native["adaptive_analysis_flag_bits"] != expected_analysis_flag_bits
      ) {
        throw worker_error("native SBS harness adaptive flag meanings differ");
      }

      const auto analysis_input = work / "analysis-input";
      // The raw per-frame adaptive trace and harness progress are transient inputs to the
      // native scene planner. Keep them below native-work so the manager's identity-pinned
      // cleanup removes them after success, failure, or cancellation. Durable bounded
      // evidence is emitted separately as scene-audit.json and the worker result contract.
      const auto analysis_output = work / "analysis-output";
      const auto cache = work / "scene-cache";
      fs::create_directories(analysis_input, ec);
      fs::create_directories(analysis_output, ec);
      if (spec.operation == "convert") {
        fs::create_directories(cache, ec);
      }
      if (ec) {
        throw worker_error("cannot create analysis directories");
      }
      streaming_decoder_t analysis_decoder {
        spec,
        media,
        work / "logs" / "analysis-decoder.log",
      };
      std::optional<streaming_decoder_t> render_decoder;
      if (spec.operation == "convert") {
        render_decoder.emplace(
          spec,
          media,
          work / "logs" / "render-decoder.log"
        );
      }
      std::vector<std::string> analysis_command {
        path_utf8(spec.sunshine_executable),
        path_utf8(spec.sunshine_config),
        "--sbs-bench",
        "--frames",
        path_utf8(analysis_input),
        "--follow",
        "--follow-format",
        analysis_decoder.format(),
        "--follow-count",
        std::to_string(media.frames.size()),
        "--out",
        path_utf8(analysis_output),
        "--artifacts",
        "adaptive",
        "--bounded-adaptive-state",
        "--observation-timeline",
        path_utf8(observation_timeline),
      };
      if (spec.operation == "convert") {
        analysis_command.insert(
          analysis_command.end(),
          {"--scene-cache", path_utf8(cache)}
        );
      }
      auto analysis = child_process_t::launch(
        analysis_command,
        spec.sunshine_executable.parent_path(),
        work / "logs" / "analysis-harness.log"
      );
      trace_tail_t trace(analysis_output, true);
      bool analysis_done = false;
      std::uint64_t live_cache_bytes = 0;
      std::uint64_t peak_cache_bytes = 0;
      std::uint64_t peak_live_raster_bytes = 0;
      std::uint64_t peak_cache_plus_raster_bytes = 0;
      std::optional<std::uint64_t> analysis_source_raster_bytes;
      std::optional<std::uint64_t> pair_bytes;
      std::uint32_t sbs_width = 0;
      std::uint32_t sbs_height = 0;
      std::vector<scene_plan_t> scenes;
      std::vector<nlohmann::json> replay_contracts;
      std::unique_ptr<whole_clip_encoder_t> conversion_encoder;
      std::unique_ptr<scene_planner_t> planner;
      nlohmann::json trace_header;
      std::uint64_t covered_until = 1;
      std::size_t accounted_scene_count = 0;
      std::size_t accounted_boundary_count = 0;
      std::size_t accounted_replay_count = 0;
      std::uintmax_t scene_audit_payload_bytes = 0;
      std::uintmax_t worker_result_payload_bytes = 0;
      nlohmann::json timeline_contract =
        spec.operation != "convert" ?
          nlohmann::json {
            {"mode", "evaluation-only"},
            {"max_output_ticks", nullptr},
          } :
          (output_container(*spec.staging_output) == "mp4" ?
             nlohmann::json {
               {"mode", "exact-rational"},
               {"max_output_ticks", 0},
               {"container", "mp4"},
             } :
             nlohmann::json {
               {"mode", "bounded-output-timebase"},
               {"max_output_ticks", 1},
               {"container", "matroska"},
               {"absolute_timestamps", true},
               {"cumulative_drift_allowed", false},
             });

      const auto account_contract_records = [&] {
        const auto boundary_count =
          planner ? planner->boundary_audit().size() : 0;
        if (
          scenes.size() > max_serialized_scene_count ||
          boundary_count > max_serialized_scene_count ||
          replay_contracts.size() > max_serialized_scene_count
        ) {
          throw worker_error(
            "scene/boundary count exceeds the bounded serialized contract"
          );
        }
        while (accounted_scene_count < scenes.size()) {
          const auto record = scene_json(scenes[accounted_scene_count++]);
          account_serialized_record(
            record,
            scene_audit_payload_bytes,
            scene_audit_max_bytes,
            "scene audit"
          );
          account_serialized_record(
            record,
            worker_result_payload_bytes,
            worker_result_max_bytes,
            "worker result"
          );
        }
        while (accounted_boundary_count < boundary_count) {
          account_serialized_record(
            boundary_json(
              planner->boundary_audit()[accounted_boundary_count++]
            ),
            scene_audit_payload_bytes,
            scene_audit_max_bytes,
            "scene audit"
          );
        }
        while (accounted_replay_count < replay_contracts.size()) {
          account_serialized_record(
            replay_contracts[accounted_replay_count++],
            worker_result_payload_bytes,
            worker_result_max_bytes,
            "worker result"
          );
        }
      };

      const auto initialize_planner =
        [&](const std::uint64_t max_open_cache_bytes) {
          if (planner || !trace_header.is_object()) {
            throw worker_error("scene planner initialization is invalid");
          }
          scene_planner_config_t planner_config;
          planner_config.max_open_cache_bytes = max_open_cache_bytes;
          planner_config.max_open_frames = default_max_open_scene_frames;
          planner_config.allow_administrative_split =
            spec.allow_administrative_split;
          planner = std::make_unique<scene_planner_t>(
            std::move(planner_config)
          );
        };

      const auto recent_scene_decisions = [&] {
        nlohmann::json decisions = nlohmann::json::array();
        const auto first = scenes.size() > 32 ? scenes.size() - 32 : 0;
        for (std::size_t index = first; index < scenes.size(); ++index) {
          decisions.push_back(scene_progress_json(scenes[index]));
        }
        return decisions;
      };

      const auto render_scenes = [&](
        const std::vector<scene_plan_t> &finalized,
        const std::uint64_t analyzed_through_sequence
      ) {
        for (const auto &scene : finalized) {
          if (scene.start_sequence != covered_until || scene.end_sequence_exclusive <= scene.start_sequence || scene.frame_count != scene.end_sequence_exclusive - scene.start_sequence) {
            throw worker_error("scene planner produced a gap/overlap");
          }
          if (scenes.size() >= max_serialized_scene_count) {
            throw worker_error(
              "scene count exceeds the bounded serialized contract"
            );
          }
          scenes.push_back(scene);
          account_contract_records();
          if (spec.operation == "convert") {
            if (!render_decoder || !sbs_width || !sbs_height) {
              throw worker_error("scene replay began before cache geometry");
            }
            publish_progress(
              spec,
              "replay",
              analyzed_through_sequence,
              &media,
              scenes.size(),
              scene_progress_json(scene),
              recent_scene_decisions()
            );
            if (!conversion_encoder) {
              conversion_encoder = std::make_unique<whole_clip_encoder_t>(
                spec,
                media,
                work,
                sbs_width,
                sbs_height
              );
            }
            const auto rendered = render_scene(
              spec,
              media,
              scene,
              *render_decoder,
              *conversion_encoder,
              work,
              cache,
              sbs_width,
              sbs_height,
              live_cache_bytes
            );
            peak_live_raster_bytes = std::max(
              peak_live_raster_bytes,
              rendered.peak_live_raster_bytes
            );
            peak_cache_plus_raster_bytes = std::max(
              peak_cache_plus_raster_bytes,
              rendered.peak_cache_plus_raster_bytes
            );
            replay_contracts.push_back({
              {"scene_id", scene.scene_id},
              {"start_sequence", scene.start_sequence},
              {"end_sequence_exclusive", scene.end_sequence_exclusive},
              {"inference_mode",
               rendered.contract.value("inference_mode", "")},
              {"depth_inference_enabled",
               rendered.contract.value("depth_inference_enabled", true)},
              {"scheduled_depth_update_count",
               rendered.contract.value(
                 "scheduled_depth_update_count",
                 1ull
               )},
              {"tensorrt_enqueue_count",
               rendered.contract.value("tensorrt_enqueue_count", 1ull)},
              {"sbs", rendered.contract.at("sbs")},
            });
            account_contract_records();
            if (scene.cache_bytes > live_cache_bytes) {
              throw worker_error("scene-cache byte ledger underflow");
            }
            live_cache_bytes -= scene.cache_bytes;
          }
          covered_until = scene.end_sequence_exclusive;
          publish_progress(
            spec,
            spec.operation == "convert" ? "analysis" : "evaluate",
            analyzed_through_sequence,
            &media,
            scenes.size(),
            nullptr,
            recent_scene_decisions()
          );
          if (is_scene_audit_checkpoint(scenes.size())) {
            write_json_atomic_bounded(
              spec.result_directory / "scene-audit.json",
              scene_audit_document(
                scenes,
                planner ? planner->boundary_audit() :
                          std::vector<boundary_audit_t> {},
                "running",
                peak_cache_bytes,
                analysis_source_raster_bytes.value_or(0),
                peak_live_raster_bytes,
                peak_cache_plus_raster_bytes,
                spec.scene_cache_hard_cap_bytes,
                timeline_contract
              ),
              scene_audit_max_bytes,
              "scene audit"
            );
          }
        }
      };

      try {
        publish_progress(spec, "analysis", 0, &media, 0);
        for (const auto &timing : media.frames) {
          const auto source = analysis_decoder.publish_next(
            analysis_input,
            timing.sequence
          );
          const auto source_raster_size = fs::file_size(source, ec);
          if (ec || source_raster_size == 0 || source_raster_size > std::numeric_limits<std::uint64_t>::max()) {
            throw worker_error("cannot measure the live analysis source raster");
          }
          const auto current_source_raster_bytes =
            static_cast<std::uint64_t>(source_raster_size);
          if (!analysis_source_raster_bytes) {
            analysis_source_raster_bytes = current_source_raster_bytes;
          } else if (*analysis_source_raster_bytes != current_source_raster_bytes) {
            throw worker_error(
              "fixed-resolution analysis source raster size changed mid-clip"
            );
          }
          if (current_source_raster_bytes > spec.scene_cache_hard_cap_bytes) {
            throw worker_error(
              "live analysis source raster exceeds the hard cap"
            );
          }
          peak_live_raster_bytes = std::max(
            peak_live_raster_bytes,
            current_source_raster_bytes
          );
          const auto progress = read_progress(
            analysis_output / "follow_progress.json",
            analysis,
            timing.sequence,
            "analysis"
          );
          if (progress.value("first_sequence", 0ull) != 1 || progress.value("last_completed_sequence", 0ull) != timing.sequence || progress.value("artifact_mode", "") != "adaptive") {
            throw worker_error("analysis global sequence ACK mismatch");
          }
          if (timing.sequence == 1) {
            trace_header = trace.read_header(analysis);
            if (spec.operation != "convert") {
              initialize_planner(std::numeric_limits<std::uint64_t>::max());
            }
          }
          std::uint64_t current_pair_bytes = 0;
          if (spec.operation == "convert") {
            const auto contract = parse_cache_contract(
              cache,
              timing.sequence,
              media
            );
            current_pair_bytes = checked_byte_sum(
              contract.depth_bytes,
              contract.state_bytes,
              "depth/state pair"
            );
            if (!pair_bytes) {
              pair_bytes = current_pair_bytes;
              sbs_width = contract.sbs_width;
              sbs_height = contract.sbs_height;
              initialize_planner(analysis_open_cache_limit(
                spec.scene_cache_hard_cap_bytes,
                current_source_raster_bytes,
                current_pair_bytes
              ));
            } else if (*pair_bytes != current_pair_bytes || sbs_width != contract.sbs_width || sbs_height != contract.sbs_height) {
              throw worker_error(
                "fixed-resolution scene-cache pair/geometry changed mid-clip"
              );
            }
            const auto after_source =
              spec.scene_cache_hard_cap_bytes - current_source_raster_bytes;
            if (current_pair_bytes > after_source || live_cache_bytes > after_source - current_pair_bytes) {
              throw worker_error(
                "scene cache plus live analysis raster exceeded the hard cap"
              );
            }
            live_cache_bytes += current_pair_bytes;
            peak_cache_bytes = std::max(peak_cache_bytes, live_cache_bytes);
          }
          const auto cache_plus_analysis_raster = checked_byte_sum(
            live_cache_bytes,
            current_source_raster_bytes,
            "analysis cache plus raster"
          );
          if (cache_plus_analysis_raster > spec.scene_cache_hard_cap_bytes) {
            throw worker_error(
              "analysis cache plus live raster exceeded the hard cap"
            );
          }
          peak_cache_plus_raster_bytes = std::max(
            peak_cache_plus_raster_bytes,
            cache_plus_analysis_raster
          );
          const auto trace_value = trace.read_frame(analysis, timing.sequence);
          remove_file_checked(source);
          if (!planner) {
            throw worker_error("scene planner was not initialized");
          }
          auto finalized = planner->feed(parse_trace_frame(
            trace_value,
            timing,
            media.time_base,
            current_pair_bytes
          ));
          account_contract_records();
          render_scenes(finalized, timing.sequence);
          publish_progress(
            spec,
            "analysis",
            timing.sequence,
            &media,
            scenes.size(),
            nullptr,
            recent_scene_decisions()
          );
        }
        publish_producer_done(analysis_input, media.frames.size());
        analysis_done = true;
        const int analysis_code = analysis.wait(
          std::chrono::duration_cast<std::chrono::milliseconds>(child_timeout)
        );
        if (analysis_code != 0) {
          throw worker_error(
            "native analysis exited " + std::to_string(analysis_code)
          );
        }
        analysis.terminate();
        analysis_decoder.finish();
        trace.finish(media.frames.size());
        if (!planner) {
          throw worker_error("scene planner received no frames");
        }
        auto finalized = planner->finish();
        account_contract_records();
        render_scenes(finalized, media.frames.size());
      } catch (const std::exception &exception) {
        if (!analysis_done) {
          publish_producer_failed(analysis_input, exception.what());
        }
        analysis.terminate();
        analysis_decoder.abort();
        if (render_decoder) {
          render_decoder->abort();
        }
        if (conversion_encoder) {
          conversion_encoder->abort(exception.what());
        }
        throw;
      }

      if (covered_until != media.frames.size() + 1) {
        throw worker_error("final scene plan does not cover every source frame");
      }
      const auto analysis_contract =
        read_json(analysis_output / "whole_clip_contract.json");
      if (analysis_contract.value("schema", 0) != 1 || analysis_contract.value("artifact_mode", "") != "adaptive" || analysis_contract.value("source_frame_count", 0ull) != media.frames.size() || analysis_contract.value("source_first_sequence", 0ull) != 1 || analysis_contract.value("inference_mode", "") != "single-pass-tensorrt" || !analysis_contract.value("depth_inference_enabled", false) || analysis_contract.value("scheduled_depth_update_count", 0ull) != media.frames.size() || analysis_contract.value("tensorrt_enqueue_count", 0ull) != media.frames.size()) {
        throw worker_error(
          "analysis did not attest exactly one TensorRT enqueue per source frame"
        );
      }
      if (
        !analysis_contract.contains("source_scope") ||
        !offline_full_frame_source_scope_is_valid(
          analysis_contract["source_scope"]
        )
      ) {
        throw worker_error(
          "analysis did not attest selected-input full-frame isolation"
        );
      }
      // Offline conversion runs the production V2 pipeline: the harness must attest the
      // depth-coordinate V2 live signed-parallax render, not cached scene geometry.
      const auto &analysis_runtime = analysis_contract.at("resolved_runtime");
      if (
        !analysis_runtime.is_object() ||
        !analysis_runtime.value("parallax_v2_render", false) ||
        !analysis_runtime.value("parallax_v2_live", false)
      ) {
        throw worker_error(
          "analysis did not attest the depth-coordinate V2 live signed-parallax render"
        );
      }
      const auto &trace_config = trace_header.at("config");
      if (
        !trace_config.is_object() ||
        trace_config.value("model", "") != analysis_runtime.value("model", "") ||
        trace_config.value("model", "") != analysis_contract.value("model", "") ||
        trace_config.value(
          "depth_reuse_interval", 0
        ) != analysis_runtime.value("depth_reuse_interval", 0) ||
        trace_config.value(
          "pop_strength", std::numeric_limits<double>::quiet_NaN()
        ) != analysis_runtime.value(
          "pop_strength", std::numeric_limits<double>::infinity()
        )
      ) {
        throw worker_error(
          "adaptive trace config disagrees with the authenticated analysis runtime"
        );
      }
      const auto &analysis_state = analysis_contract.at("adaptive_state");
      if (
        !analysis_state.is_object() ||
        analysis_state.size() != 7u ||
        analysis_state.value("transport", "") != "atomic-latest-v1" ||
        analysis_state.value("header_file", "") != "adaptive_state_header.json" ||
        analysis_state.value("frame_file", "") != "adaptive_state_frame.json" ||
        analysis_state.value("retained_history", true) ||
        analysis_state.value("schema", 0u) != sbs_adaptive_state::schema_version ||
        analysis_state.value("capture", "") != sbs_adaptive_state::capture ||
        analysis_state.value("frame_count", 0ull) != media.frames.size() ||
        analysis_contract.contains("cut_state")
      ) {
        throw worker_error(
          "analysis did not attest its bounded adaptive-state transport"
        );
      }
      if (spec.operation == "convert") {
        if (!render_decoder) {
          throw worker_error("conversion lacks its render decoder");
        }
        render_decoder->finish();
        if (!conversion_encoder) {
          throw worker_error("conversion produced no whole-clip encoder");
        }
        conversion_encoder->finish();
        if (live_cache_bytes != 0) {
          throw worker_error("scene cache was not empty after final replay");
        }
        std::uintmax_t cache_media_files = 0;
        for (const auto &entry : fs::directory_iterator(cache, ec)) {
          if (entry.is_regular_file()) {
            const auto extension = lower(entry.path().extension().string());
            if (extension == ".r32f" || extension == ".u32") {
              ++cache_media_files;
            }
          }
        }
        if (cache_media_files != 0) {
          throw worker_error("scene cache retained depth/state artifacts");
        }
        publish_progress(
          spec,
          "mux",
          media.frames.size(),
          &media,
          scenes.size(),
          nullptr,
          recent_scene_decisions()
        );
        {
          // Release the pre-mux timing vector before probing the final container. The source
          // timeline remains the sole long-lived compact vector; verification never retains
          // source + encoded + output frame inventories simultaneously.
          const auto encoded_media = probe_media(
            spec,
            conversion_encoder->encoded_video(),
            work / "encoded-video-probe.json",
            work / "logs" / "encoded-video-probe.log",
            false
          );
          validate_timeline_contract(media, encoded_media, 0);
          if (encoded_media.codec_name != codec_name_for_request(spec.codec) || encoded_media.width != sbs_width || encoded_media.height != sbs_height) {
            throw worker_error(
              "whole-clip compressed video codec/raster mismatch"
            );
          }
          if (media.color != media_color_e::sdr) {
            validate_hdr_equivalence(media, encoded_media);
          } else {
            validate_sdr_color_equivalence(media, encoded_media);
          }
        }
        mux_encoded_video(
          spec,
          media,
          *source_inventory,
          conversion_encoder->encoded_video(),
          work / "logs" / "final-mux.log"
        );
        const bool matroska_output =
          output_container(*spec.staging_output) == "matroska";
        {
          const auto output_media = probe_media(
            spec,
            *spec.staging_output,
            work / "output-probe.json",
            work / "logs" / "output-probe.log",
            false
          );
          if (matroska_output) {
            require_millisecond_or_finer(
              output_media.time_base,
              "output video"
            );
          }
          validate_timeline_contract(
            media,
            output_media,
            matroska_output ? 1 : 0
          );
          timeline_contract["observed_video"] =
            timeline_error_report(media, output_media);
          if (output_media.codec_name != codec_name_for_request(spec.codec) || output_media.width != sbs_width || output_media.height != sbs_height) {
            throw worker_error("final output codec/raster contract mismatch");
          }
          if (media.color != media_color_e::sdr) {
            validate_hdr_equivalence(media, output_media);
          } else {
            validate_sdr_color_equivalence(media, output_media);
          }
          write_json_atomic(
            spec.result_directory / "output-contract.json",
            media_contract_json(output_media)
          );
        }
        const auto output_inventory = probe_stream_inventory(
          spec,
          *spec.staging_output,
          work / "output-stream-inventory.json",
          work / "logs" / "output-stream-inventory.log"
        );
        timeline_contract["observed_auxiliary"] =
          validate_stream_inventory(
            *source_inventory,
            output_inventory,
            matroska_output ? 1 : 0
          );
        write_json_atomic(
          spec.result_directory / "timeline-contract.json",
          timeline_contract
        );
      }

      account_contract_records();
      const auto audit = scene_audit_document(
        scenes,
        planner->boundary_audit(),
        "complete",
        peak_cache_bytes,
        analysis_source_raster_bytes.value_or(0),
        peak_live_raster_bytes,
        peak_cache_plus_raster_bytes,
        spec.scene_cache_hard_cap_bytes,
        timeline_contract
      );
      write_json_atomic_bounded(
        spec.result_directory / "scene-audit.json",
        audit,
        scene_audit_max_bytes,
        "scene audit"
      );
      publish_progress(
        spec,
        "complete",
        media.frames.size(),
        &media,
        scenes.size(),
        nullptr,
        recent_scene_decisions()
      );
      const auto staging_identity = staging_claim ?
                                      staging_claim->identity_json() :
                                      nlohmann::json(nullptr);
      nlohmann::json result {
        {"schema", 1},
        {"job_id", spec.job_id},
        {"status", "complete"},
        {"operation", spec.operation},
        {"codec", spec.codec},
        {"worker_spec_sha256", spec.authenticated_spec_sha256},
        {"python_dependency", false},
        {"source", {
                     {"width", media.width},
                     {"height", media.height},
                     {"frame_count", media.frames.size()},
                     {"duration_seconds", media.duration_seconds()},
                     {"variable_frame_rate", media.variable_frame_rate()},
                     {"color_mode", media.color == media_color_e::sdr ? "sdr" : (media.color == media_color_e::hdr_pq ? "hdr-pq" : "hdr-hlg")},
                     {"contract", path_utf8(spec.result_directory / "source-contract.json")},
                   }},
        {"scene_count", scenes.size()},
        {"scenes", [&] {
           nlohmann::json values = nlohmann::json::array();
           for (const auto &scene : scenes) {
             values.push_back(scene_json(scene));
           }
           return values;
         }()},
        {"scene_audit", path_utf8(spec.result_directory / "scene-audit.json")},
        {"analysis_contract", analysis_contract},
        {"replay_contracts", replay_contracts},
        {"cache", {
                    {"hard_cap_bytes", spec.scene_cache_hard_cap_bytes},
                    {"peak_bytes", peak_cache_bytes},
                    {"analysis_source_raster_bytes", analysis_source_raster_bytes.value_or(0)},
                    {"peak_live_raster_bytes", peak_live_raster_bytes},
                    {"peak_cache_plus_raster_bytes", peak_cache_plus_raster_bytes},
                    {"remaining_bytes", live_cache_bytes},
                  }},
        {"timeline_contract", timeline_contract},
        {"staging_identity", std::move(staging_identity)},
        {"output", spec.staging_output ? nlohmann::json(path_utf8(*spec.staging_output)) : nlohmann::json(nullptr)},
      };
      write_json_atomic_bounded(
        spec.result_path,
        result,
        worker_result_max_bytes,
        "worker result"
      );
      // The manager retains the job-root identity handle and removes native-work
      // only after this process (and its process group) has been reaped. A child
      // cannot safely re-resolve or delete that user-writable pathname, and on
      // Windows such an attempt also conflicts with the manager's delete pin.
      if (staging_claim) {
        staging_claim->release_for_publish();
      }
      return 0;
    } catch (const std::exception &exception) {
      if (parsed_spec) {
        publish_failure(*parsed_spec, exception.what());
      }
      return 2;
    }
  }
}  // namespace offline_sbs

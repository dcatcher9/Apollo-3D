/**
 * @file src/offline_sbs_worker_probe.cpp
 * @brief FFprobe media contracts: strict bounded probing of source, intermediate and output
 * media, exact rational timeline checks, and the auxiliary stream inventory.
 */

#include "offline_sbs_worker_probe.h"

#include "offline_sbs_wire_contract.h"
#include "offline_sbs_worker_io.h"

#include <algorithm>
#include <array>
#include <boost/multiprecision/cpp_int.hpp>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <openssl/evp.h>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace offline_sbs {
  namespace fs = std::filesystem;
  using namespace std::chrono_literals;
  using namespace worker_detail;

  namespace {
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

    std::int64_t positive_tick_difference(
      std::int64_t later,
      std::int64_t earlier,
      std::string_view description
    );

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

    static_assert(
      max_auxiliary_packets * sizeof(packet_timing_t) <=
        max_retained_packet_bytes,
      "the auxiliary packet vector exceeds its retained-memory contract"
    );

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
              "source contains a presentation gap/overlap which the re-encoded "
              "output cannot reproduce exactly"
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
            "SDR video side data cannot be preserved through the raw frame pipe"
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

  namespace worker_detail {
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

    // A Matroska output stores millisecond timestamps, so its final probe allows one output
    // tick per frame. The intermediate MP4 keeps the source time base: its packets prove the
    // encoder timeline exactly without decoding the stream that the final probe decodes.
    void probe_encoded_packet_timeline(
      const worker_spec_t &spec,
      const media_contract_t &source,
      const fs::path &encoded_video,
      const fs::path &log_path
    ) {
      auto metadata_log = log_path;
      metadata_log.replace_filename(
        log_path.stem().string() + "-metadata" + log_path.extension().string()
      );
      const auto metadata = run_logged_json_bounded(
        build_ffprobe_metadata_command(spec, encoded_video),
        spec.sunshine_executable.parent_path(),
        metadata_log,
        max_video_metadata_probe_bytes,
        "FFprobe encoded video metadata"
      );
      if (
        !metadata.is_object() ||
        !metadata.contains("streams") ||
        !metadata["streams"].is_array() ||
        metadata["streams"].size() != 1 ||
        !metadata["streams"][0].is_object() ||
        !metadata["streams"][0].contains("time_base")
      ) {
        throw worker_error(
          "FFprobe must return exactly one encoded video stream descriptor"
        );
      }
      const auto time_base =
        parse_rational(metadata["streams"][0]["time_base"], "encoded time_base");
      const auto deadline = std::chrono::steady_clock::now() + child_timeout;
      auto child = child_process_t::launch(
        {
          path_utf8(spec.ffprobe_executable),
          "-v",
          "error",
          "-select_streams",
          "v:0",
          "-show_entries",
          "packet=pts,duration",
          "-of",
          "csv=p=0",
          "-protocol_whitelist",
          "file",
          path_utf8(encoded_video),
        },
        spec.sunshine_executable.parent_path(),
        log_path,
        std::nullopt,
        true
      );
      child_stdout_streambuf_t stdout_buffer(child, deadline);
      std::istream stdout_stream(&stdout_buffer);
      stdout_stream.exceptions(std::ios::badbit);
      validate_encoded_packet_timeline(source, time_base, stdout_stream);
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
  }  // namespace worker_detail

#ifdef SUNSHINE_TESTS
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

#ifdef SUNSHINE_TESTS
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
#endif

#ifdef SUNSHINE_TESTS
  void validate_timeline_equivalence(
    const media_contract_t &source,
    const media_contract_t &actual
  ) {
    validate_timeline_contract(source, actual, 1);
  }
#endif

  void validate_encoded_packet_timeline(
    const media_contract_t &source,
    const rational_t time_base,
    std::istream &packets
  ) {
    constexpr std::size_t max_packet_line_bytes = 4096;
    const auto parse_field = [](const std::string_view text) {
      std::int64_t value = 0;
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
      if (parsed.ec != std::errc {} || parsed.ptr != text.data() + text.size()) {
        throw worker_error("FFprobe encoded packet lacks an exact timestamp or duration");
      }
      return value;
    };
    std::size_t index = 0;
    std::string line;
    for (auto character = packets.get(); character != std::char_traits<char>::eof();) {
      line.clear();
      while (character != std::char_traits<char>::eof() && character != '\n') {
        if (line.size() == max_packet_line_bytes) {
          throw worker_error("FFprobe encoded packet line exceeds its bounded contract");
        }
        line.push_back(static_cast<char>(character));
        character = packets.get();
      }
      if (character == '\n') {
        character = packets.get();
      }
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      const auto pts_end = line.find(',');
      if (pts_end == std::string::npos) {
        throw worker_error("FFprobe encoded packet line is malformed");
      }
      const auto duration_end = std::min(line.find(',', pts_end + 1), line.size());
      const std::string_view fields {line};
      const auto pts = parse_field(fields.substr(0, pts_end));
      const auto duration =
        parse_field(fields.substr(pts_end + 1, duration_end - pts_end - 1));
      if (index == source.frames.size()) {
        throw worker_error("encoded frame count differs from the source timeline");
      }
      const auto &wanted = source.frames[index++];
      if (!rational_ticks_within_actual_ticks(wanted.pts, source.time_base, pts, time_base, 0) || !rational_ticks_within_actual_ticks(wanted.duration, source.time_base, duration, time_base, 0)) {
        throw worker_error(
          "encoded presentation timeline differs at frame " + std::to_string(index)
        );
      }
    }
    if (index != source.frames.size()) {
      throw worker_error("encoded frame count differs from the source timeline");
    }
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
}  // namespace offline_sbs

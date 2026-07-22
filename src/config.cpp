/**
 * @file src/config.cpp
 * @brief Definitions for the configuration of Sunshine.
 */
// standard includes
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <utility>

// lib includes
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

// local includes
#include "config.h"
#include "entry_handler.h"
#include "file_handler.h"
#include "logging.h"
#include "network.h"
#include "nvhttp.h"
#include "platform/common.h"
#include "rtsp.h"
#include "stream.h"
#include "utility.h"
#include "video.h"

#ifdef _WIN32
  #include "platform/windows/utils.h"

  #include <shellapi.h>
#endif

namespace fs = std::filesystem;
using namespace std::literals;

#define CA_DIR "credentials"
#define PRIVATE_KEY_FILE CA_DIR "/cakey.pem"
#define CERTIFICATE_FILE CA_DIR "/cacert.pem"

#define APPS_JSON_PATH platf::appdata().string() + "/apps.json"

namespace config {

  namespace nv {

    nvenc::nvenc_two_pass twopass_from_view(const ::std::string_view &preset) {
      if (preset == "disabled") {
        return nvenc::nvenc_two_pass::disabled;
      }
      if (preset == "quarter_res") {
        return nvenc::nvenc_two_pass::quarter_resolution;
      }
      if (preset == "full_res") {
        return nvenc::nvenc_two_pass::full_resolution;
      }
      BOOST_LOG(warning) << "config: unknown nvenc_twopass value: " << preset;
      return nvenc::nvenc_two_pass::quarter_resolution;
    }

  }  // namespace nv

  video_t video {
    {},  // nv
    true,  // nv_realtime_hags
    true,  // nv_opengl_vulkan_on_dxgi
    true,  // nv_sunshine_high_power_mode
    {},  // adapter_name
    {},  // output_name

    0,  // max_bitrate
    0,  // minimum_fps_target (0 = framerate)

    {},  // sbs (tuned defaults are the sbs_t member initializers in config.h)
  };

  audio_t audio {
    {},  // audio_sink
    {},  // virtual_sink
  };

  stream_t stream {
    10s,  // ping_timeout
    60s,  // session_resume_grace

    APPS_JSON_PATH,

    20,  // fecPercentage

    0,  // packet_size_limit
  };

  nvhttp_t nvhttp {
    "lan",  // origin web manager

    PRIVATE_KEY_FILE,
    CERTIFICATE_FILE,

    platf::get_host_name(),  // sunshine_name,
    "sunshine_state.json"s,  // file_state
  };

  input_t input {
    {
      {0x10, 0xA0},
      {0x11, 0xA2},
      {0x12, 0xA4},
    },
    -1ms,  // back_button_timeout
    500ms,  // key_repeat_delay
    std::chrono::duration<double> {1 / 24.9},  // key_repeat_period

    {
      platf::supported_gamepads(nullptr).front().name.data(),
      platf::supported_gamepads(nullptr).front().name.size(),
    },  // Default gamepad
    true,  // back as touchpad click enabled (manual DS4 only)
    true,  // client gamepads with motion events are emulated as DS4
    true,  // client gamepads with touchpads are emulated as DS4

    true,  // always send scancodes
    true,  // high resolution scrolling
    true,  // native pen/touch support
    true,  // forward_rumble
  };

  sunshine_t sunshine {
    false,  // hide_tray_controls
    true,  // enable_pairing
    true,  // enable_discovery
    "en",  // locale
    2,  // min_log_level
    false,  // diagnostics_enabled
    0,  // flags
    {},  // User file
    {},  // Username
    {},  // Password
    {},  // Password Salt
    platf::appdata().string() + "/sunshine.conf",  // config file
    {},  // cmd args
    47989,  // Base port number
    "ipv4",  // Address family
    {},  // Bind address (empty = all interfaces)
    platf::appdata().string() + "/sunshine.log",  // log file
    true,  // system_tray
    {},  // prep commands
  };

  const std::vector<depth_model_info> &depth_model_registry() {
    // Built-in DA-V2 model definitions referenced by profile depth_model names.
    static const std::vector<depth_model_info> registry = {
      {"depth_anything_v2_fp16",
       "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx"},
      {"depth_anything_v2_base_fp16",
       "https://huggingface.co/onnx-community/depth-anything-v2-base/resolve/main/onnx/model_fp16.onnx"},
    };
    return registry;
  }

  bool endline(char ch) {
    return ch == '\r' || ch == '\n';
  }

  bool space_tab(char ch) {
    return ch == ' ' || ch == '\t';
  }

  bool whitespace(char ch) {
    return space_tab(ch) || endline(ch);
  }

  std::string to_string(const char *begin, const char *end) {
    std::string result;

    KITTY_WHILE_LOOP(auto pos = begin, pos != end, {
      auto comment = std::find(pos, end, '#');
      auto endl = std::find_if(comment, end, endline);

      result.append(pos, comment);

      pos = endl;
    })

    return result;
  }

  template<class It>
  It skip_list(It skipper, It end) {
    int stack = 1;
    while (skipper != end && stack) {
      if (*skipper == '[') {
        ++stack;
      }
      if (*skipper == ']') {
        --stack;
      }

      ++skipper;
    }

    return skipper;
  }

  std::pair<
    ::std::string_view::const_iterator,
    ::std::optional<std::pair<std::string, std::string>>>
    parse_option(::std::string_view::const_iterator begin, ::std::string_view::const_iterator end) {
    begin = std::find_if_not(begin, end, whitespace);
    auto endl = std::find_if(begin, end, endline);
    // Inline comments: '#' starts a comment at line start or when preceded by whitespace,
    // so `key = value  # note` works while values containing '#' (URLs, fragments) survive.
    auto endc = std::find(begin, endl, '#');
    while (endc != endl && endc != begin && !whitespace(*(endc - 1))) {
      endc = std::find(endc + 1, endl, '#');
    }
    endc = std::find_if(std::make_reverse_iterator(endc), std::make_reverse_iterator(begin), std::not_fn(whitespace)).base();

    auto eq = std::find(begin, endc, '=');
    if (eq == endc || eq == begin) {
      return std::make_pair(endl, std::nullopt);
    }

    auto end_name = std::find_if_not(std::make_reverse_iterator(eq), std::make_reverse_iterator(begin), space_tab).base();
    auto begin_val = std::find_if_not(eq + 1, endc, space_tab);

    if (begin_val == endc) {
      // Empty value (or nothing but an inline comment after '=').
      return std::make_pair(endl, std::nullopt);
    }

    // The value ends at the inline comment (whitespace-trimmed); lists extend to their ']'.
    auto val_end = endc;

    // Lists might contain newlines
    if (*begin_val == '[') {
      endl = skip_list(begin_val + 1, end);

      // Check if we reached the end of the file without finding a closing bracket
      // We know we have a valid closing bracket if:
      // 1. We didn't reach the end, or
      // 2. We reached the end but the last character was the matching closing bracket
      if (endl == end && end == begin_val + 1) {
        BOOST_LOG(warning) << "config: Missing ']' in config option: " << to_string(begin, end_name);
        return std::make_pair(endl, std::nullopt);
      }
      val_end = endl;
    }

    return std::make_pair(
      endl,
      std::make_pair(to_string(begin, end_name), to_string(begin_val, val_end))
    );
  }

  std::unordered_map<std::string, std::string> parse_config(const ::std::string_view &file_content) {
    std::unordered_map<std::string, std::string> vars;

    auto pos = std::begin(file_content);
    auto end = std::end(file_content);

    while (pos < end) {
      // auto newline = std::find_if(pos, end, [](auto ch) { return ch == '\n' || ch == '\r'; });
      TUPLE_2D(endl, var, parse_option(pos, end));

      pos = endl;
      if (pos != end) {
        pos += (*pos == '\r') ? 2 : 1;
      }

      if (!var) {
        continue;
      }

      vars.emplace(std::move(*var));
    }

    return vars;
  }

  void string_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, std::string &input) {
    auto it = vars.find(name);
    if (it == std::end(vars)) {
      return;
    }

    input = std::move(it->second);

    vars.erase(it);
  }

  template<typename T, typename F>
  void generic_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, T &input, F &&f) {
    std::string tmp;
    string_f(vars, name, tmp);
    if (!tmp.empty()) {
      input = f(tmp);
    }
  }

  void string_restricted_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, std::string &input, const std::vector<::std::string_view> &allowed_vals) {
    std::string temp;
    string_f(vars, name, temp);

    for (auto &allowed_val : allowed_vals) {
      if (temp == allowed_val) {
        input = std::move(temp);
        return;
      }
    }
  }

  void path_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, fs::path &input) {
    // appdata needs to be retrieved once only
    static auto appdata = platf::appdata();

    std::string temp;
    string_f(vars, name, temp);

    if (!temp.empty()) {
      input = temp;
    }

    if (input.is_relative()) {
      input = appdata / input;
    }

    auto dir = input;
    dir.remove_filename();

    // Ensure the directories exists
    if (!fs::exists(dir)) {
      fs::create_directories(dir);
    }
  }

  void path_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, std::string &input) {
    fs::path temp = input;

    path_f(vars, name, temp);

    input = temp.string();
  }

  void int_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, int &input) {
    auto it = vars.find(name);

    if (it == std::end(vars)) {
      return;
    }

    ::std::string_view val = it->second;

    // If value is something like: "756" instead of 756
    if (val.size() >= 2 && val[0] == '"') {
      val = val.substr(1, val.size() - 2);
    }

    // If that integer is in hexadecimal
    if (val.size() >= 2 && val.substr(0, 2) == "0x"sv) {
      input = util::from_hex<int>(val.substr(2));
    } else {
      input = util::from_view(val);
    }

    vars.erase(it);
  }

  void int_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, ::std::optional<int> &input) {
    auto it = vars.find(name);

    if (it == std::end(vars)) {
      return;
    }

    ::std::string_view val = it->second;

    // If value is something like: "756" instead of 756
    if (val.size() >= 2 && val[0] == '"') {
      val = val.substr(1, val.size() - 2);
    }

    // If that integer is in hexadecimal
    if (val.size() >= 2 && val.substr(0, 2) == "0x"sv) {
      input = util::from_hex<int>(val.substr(2));
    } else {
      input = util::from_view(val);
    }

    vars.erase(it);
  }

  template<class F>
  void int_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, int &input, F &&f) {
    std::string tmp;
    string_f(vars, name, tmp);
    if (!tmp.empty()) {
      input = f(tmp);
    }
  }

  template<class F>
  void int_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, ::std::optional<int> &input, F &&f) {
    std::string tmp;
    string_f(vars, name, tmp);
    if (!tmp.empty()) {
      input = f(tmp);
    }
  }

  void int_between_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, int &input, const std::pair<int, int> &range) {
    int temp = input;

    int_f(vars, name, temp);

    TUPLE_2D_REF(lower, upper, range);
    if (temp >= lower && temp <= upper) {
      input = temp;
    }
  }

  bool to_bool(std::string &boolean) {
    std::for_each(std::begin(boolean), std::end(boolean), [](char ch) {
      return (char) std::tolower(ch);
    });

    return boolean == "true"sv ||
           boolean == "yes"sv ||
           boolean == "enable"sv ||
           boolean == "enabled"sv ||
           boolean == "on"sv ||
           (std::find(std::begin(boolean), std::end(boolean), '1') != std::end(boolean));
  }

  void bool_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, bool &input) {
    std::string tmp;
    string_f(vars, name, tmp);

    if (tmp.empty()) {
      return;
    }

    input = to_bool(tmp);
  }

  void double_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, double &input) {
    std::string tmp;
    string_f(vars, name, tmp);

    if (tmp.empty()) {
      return;
    }

    char *c_str_p;
    auto val = std::strtod(tmp.c_str(), &c_str_p);

    if (c_str_p == tmp.c_str()) {
      return;
    }

    input = val;
  }

  void double_between_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, double &input, const std::pair<double, double> &range) {
    double temp = input;

    double_f(vars, name, temp);

    TUPLE_2D_REF(lower, upper, range);
    if (temp >= lower && temp <= upper) {
      input = temp;
    }
  }

  void list_string_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, std::vector<std::string> &input) {
    std::string string;
    string_f(vars, name, string);

    if (string.empty()) {
      return;
    }

    input.clear();

    auto begin = std::cbegin(string);
    if (*begin == '[') {
      ++begin;
    }

    begin = std::find_if_not(begin, std::cend(string), whitespace);
    if (begin == std::cend(string)) {
      return;
    }

    auto pos = begin;
    while (pos < std::cend(string)) {
      if (*pos == '[') {
        pos = skip_list(pos + 1, std::cend(string)) + 1;
      } else if (*pos == ']') {
        break;
      } else if (*pos == ',') {
        input.emplace_back(begin, pos);
        pos = begin = std::find_if_not(pos + 1, std::cend(string), whitespace);
      } else {
        ++pos;
      }
    }

    if (pos != begin) {
      input.emplace_back(begin, pos);
    }
  }

  void list_prep_cmd_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, std::vector<prep_cmd_t> &input) {
    std::string string;
    string_f(vars, name, string);

    std::stringstream jsonStream;

    // check if string is empty, i.e. when the value doesn't exist in the config file
    if (string.empty()) {
      return;
    }

    // We need to add a wrapping object to make it valid JSON, otherwise ptree cannot parse it.
    jsonStream << "{\"prep_cmd\":" << string << "}";

    boost::property_tree::ptree jsonTree;
    boost::property_tree::read_json(jsonStream, jsonTree);

    for (auto &[_, prep_cmd] : jsonTree.get_child("prep_cmd"s)) {
      auto do_cmd = prep_cmd.get_optional<std::string>("do"s);
      auto undo_cmd = prep_cmd.get_optional<std::string>("undo"s);
      auto elevated = prep_cmd.get_optional<bool>("elevated"s);

      input.emplace_back(do_cmd.value_or(""), undo_cmd.value_or(""), elevated.value_or(false));
    }
  }

  void list_int_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, std::vector<int> &input) {
    std::vector<std::string> list;
    list_string_f(vars, name, list);

    // check if list is empty, i.e. when the value doesn't exist in the config file
    if (list.empty()) {
      return;
    }

    // The framerate list must be cleared before adding values from the file configuration.
    // If the list is not cleared, then the specified parameters do not affect the behavior of the sunshine server.
    // That is, if you set only 30 fps in the configuration file, it will not work because by default, during initialization the list includes 10, 30, 60, 90 and 120 fps.
    input.clear();
    for (auto &el : list) {
      ::std::string_view val = el;

      // If value is something like: "756" instead of 756
      if (val.size() >= 2 && val[0] == '"') {
        val = val.substr(1, val.size() - 2);
      }

      int tmp;

      // If the integer is a hexadecimal
      if (val.size() >= 2 && val.substr(0, 2) == "0x"sv) {
        tmp = util::from_hex<int>(val.substr(2));
      } else {
        tmp = util::from_view(val);
      }
      input.emplace_back(tmp);
    }
  }

  void map_int_int_f(std::unordered_map<std::string, std::string> &vars, const std::string &name, std::unordered_map<int, int> &input) {
    std::vector<int> list;
    list_int_f(vars, name, list);

    // The list needs to be a multiple of 2
    if (list.size() % 2) {
      BOOST_LOG(warning) << "config: expected "sv << name << " to have a multiple of two elements --> not "sv << list.size();
      return;
    }

    int x = 0;
    while (x < list.size()) {
      auto key = list[x++];
      auto val = list[x++];

      input.emplace(key, val);
    }
  }

  int apply_flags(const char *line) {
    int ret = 0;
    while (*line != '\0') {
      switch (*line) {
        case '1':
          config::sunshine.flags[config::flag::FRESH_STATE].flip();
          break;
        default:
          BOOST_LOG(warning) << "config: Unrecognized flag: ["sv << *line << ']' << std::endl;
          ret = -1;
      }

      ++line;
    }

    return ret;
  }

  std::vector<::std::string_view> &get_supported_gamepad_options() {
    const auto options = platf::supported_gamepads(nullptr);
    static std::vector<::std::string_view> opts {};
    opts.reserve(options.size());
    for (auto &opt : options) {
      opts.emplace_back(opt.name);
    }
    return opts;
  }

  void apply_config(std::unordered_map<std::string, std::string> &&vars) {
    for (auto &[name, val] : vars) {
#ifdef _WIN32
      BOOST_LOG(info) << "config: ["sv << name << "] -- ["sv << utf8ToAcp(val) << ']';
#else
      BOOST_LOG(info) << "config: ["sv << name << "] -- ["sv << val << ']';
#endif
      modified_config_settings[name] = val;
    }

    int_between_f(vars, "nvenc_preset", video.nv.quality_preset, {1, 7});
    int_between_f(vars, "nvenc_vbv_increase", video.nv.vbv_percentage_increase, {0, 400});
    bool_f(vars, "nvenc_spatial_aq", video.nv.adaptive_quantization);
    bool_f(vars, "nvenc_hevc_unidirectional_b", video.nv.hevc_unidirectional_b);
    generic_f(vars, "nvenc_twopass", video.nv.two_pass, nv::twopass_from_view);
    bool_f(vars, "nvenc_realtime_hags", video.nv_realtime_hags);
    bool_f(vars, "nvenc_opengl_vulkan_on_dxgi", video.nv_opengl_vulkan_on_dxgi);
    bool_f(vars, "nvenc_latency_over_power", video.nv_sunshine_high_power_mode);

    string_f(vars, "adapter_name", video.adapter_name);
    string_f(vars, "output_name", video.output_name);

    int_f(vars, "max_bitrate", video.max_bitrate);
    double_between_f(vars, "minimum_fps_target", video.minimum_fps_target, {0.0, 1000.0});

    // Apply one complete startup profile first. Individual sbs_3d_* keys are parsed afterwards,
    // so an explicitly configured parameter always overrides its profile value. Reinitializing
    // the struct also clears stale values when a config reload removes an override.
    std::string sbs_profile = "apollo";
    string_f(vars, "sbs_3d_profile", sbs_profile);
    if (sbs_profile.empty() || sbs_profile.size() > 64 || sbs_profile.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos) {
      BOOST_LOG(warning) << "Invalid sbs_3d_profile name '" << sbs_profile
                         << "'; use 1-64 letters, digits, '_' or '-'. Using 'apollo'.";
      sbs_profile = "apollo";
    }
    if (sbs_profile == "vd3d") {
      BOOST_LOG(warning) << "SBS profile 'vd3d' was retired; using 'apollo'.";
      sbs_profile = "apollo";
    }
    auto apply_sbs_values = [&](video_t::sbs_t &target, const std::string &prefix) {
      double_between_f(vars, prefix + "pop_strength", target.pop_strength, {0.25, 2.0});
      bool_f(vars, prefix + "adaptive_pop", target.adaptive_pop);
      double_between_f(vars, prefix + "adaptive_pop_max", target.adaptive_pop_max, {0.25, 2.0});
      double_between_f(vars, prefix + "ema", target.ema, {0.01, 1.0});
      double_between_f(vars, prefix + "ema_edge_change", target.ema_edge_change, {0.0, 1.0});
      double_between_f(vars, prefix + "ema_edge_gradient", target.ema_edge_gradient, {0.0, 1.0});
      double_between_f(vars, prefix + "ema_edge_strength", target.ema_edge_strength, {0.0, 1.0});
      int_f(vars, prefix + "depth_short_side", target.depth_short_side);
      double_between_f(vars, prefix + "depth_max_aspect", target.depth_max_aspect, {1.0, 8.0});
      double_between_f(vars, prefix + "minmax_ema", target.minmax_ema, {0.001, 1.0});
      double_between_f(vars, prefix + "subject_lock", target.subject_lock, {0.0, 1.0});
      double_between_f(vars, prefix + "subject_recenter", target.subject_recenter, {0.0, 1.0});
      bool_f(vars, prefix + "subject_stretch", target.subject_stretch);
      string_f(vars, prefix + "zero_plane", target.zero_plane);
      string_f(vars, prefix + "depth_model", target.depth_model);
      string_f(vars, prefix + "depth_model_url", target.depth_model_url);
      int_between_f(vars, prefix + "max_encode_width", target.max_encode_width, {256, 16384});
      bool_f(vars, prefix + "cuda_graph", target.cuda_graph);
    };

    video.sbs = {};
    video.sbs.profile = sbs_profile;
    apply_sbs_values(video.sbs, "sbs_3d_profile_" + sbs_profile + "_");
    apply_sbs_values(video.sbs, "sbs_3d_");
    if (video.sbs.zero_plane != "legacy" && video.sbs.zero_plane != "subject" && video.sbs.zero_plane != "median" && video.sbs.zero_plane != "background") {
      BOOST_LOG(warning) << "Invalid sbs_3d_zero_plane value '" << video.sbs.zero_plane
                         << "'; use legacy, subject, median, or background. Using legacy.";
      video.sbs.zero_plane = "legacy";
    }
    video.sbs.max_encode_width &= ~1;

    path_f(vars, "pkey", nvhttp.pkey);
    path_f(vars, "cert", nvhttp.cert);
    string_f(vars, "sunshine_name", nvhttp.sunshine_name);
    path_f(vars, "log_path", config::sunshine.log_file);
    path_f(vars, "file_state", nvhttp.file_state);

    // Must be run after "file_state"
    config::sunshine.credentials_file = config::nvhttp.file_state;
    path_f(vars, "credentials_file", config::sunshine.credentials_file);

    list_prep_cmd_f(vars, "global_prep_cmd", config::sunshine.prep_cmds);

    string_f(vars, "audio_sink", audio.sink);
    string_f(vars, "virtual_sink", audio.virtual_sink);

    string_restricted_f(vars, "origin_web_ui_allowed", nvhttp.origin_web_ui_allowed, {"pc"sv, "lan"sv, "wan"sv});

    int to = -1;
    int_between_f(vars, "ping_timeout", to, {-1, std::numeric_limits<int>::max()});
    if (to != -1) {
      stream.ping_timeout = std::chrono::milliseconds(to);
    }

    int session_resume_grace = -1;
    int_between_f(vars, "session_resume_grace", session_resume_grace, {0, 600000});
    if (session_resume_grace != -1) {
      stream.session_resume_grace = std::chrono::milliseconds(session_resume_grace);
    }

    {
      int packet_size_limit = 0;
      int_f(vars, "packetsize", packet_size_limit);
      if (packet_size_limit == 0 || ::stream::is_valid_video_packet_size(packet_size_limit)) {
        stream.packet_size_limit = packet_size_limit;
      } else {
        BOOST_LOG(warning) << "Ignoring invalid packetsize limit ["sv << packet_size_limit
                           << "]; use 0 or "sv << ::stream::VIDEO_PACKET_SIZE_MIN << '-'
                           << ::stream::VIDEO_PACKET_SIZE_MAX << '.';
        stream.packet_size_limit = 0;
      }
    }

    path_f(vars, "file_apps", stream.file_apps);
#ifndef __ANDROID__
    // TODO: Android can possibly support this
    if (!fs::exists(stream.file_apps.c_str())) {
      fs::copy_file(SUNSHINE_ASSETS_DIR "/apps.json", stream.file_apps);
      fs::permissions(
        stream.file_apps,
        fs::perms::owner_read | fs::perms::owner_write,
        fs::perm_options::add
      );
    }
#endif

    int_between_f(vars, "fec_percentage", stream.fec_percentage, {1, 255});

    map_int_int_f(vars, "keybindings"s, input.keybindings);

    to = std::numeric_limits<int>::min();
    int_f(vars, "back_button_timeout", to);

    if (to > std::numeric_limits<int>::min()) {
      input.back_button_timeout = std::chrono::milliseconds {to};
    }

    double repeat_frequency {0};
    double_between_f(vars, "key_repeat_frequency", repeat_frequency, {0, std::numeric_limits<double>::max()});

    if (repeat_frequency > 0) {
      config::input.key_repeat_period = std::chrono::duration<double> {1 / repeat_frequency};
    }

    to = -1;
    int_f(vars, "key_repeat_delay", to);
    if (to >= 0) {
      input.key_repeat_delay = std::chrono::milliseconds {to};
    }

    string_restricted_f(vars, "gamepad"s, input.gamepad, get_supported_gamepad_options());
    bool_f(vars, "ds4_back_as_touchpad_click", input.ds4_back_as_touchpad_click);
    bool_f(vars, "motion_as_ds4", input.motion_as_ds4);
    bool_f(vars, "touchpad_as_ds4", input.touchpad_as_ds4);

    bool_f(vars, "always_send_scancodes", input.always_send_scancodes);

    bool_f(vars, "high_resolution_scrolling", input.high_resolution_scrolling);
    bool_f(vars, "native_pen_touch", input.native_pen_touch);
    bool_f(vars, "system_tray", sunshine.system_tray);
    bool_f(vars, "hide_tray_controls", sunshine.hide_tray_controls);
    bool_f(vars, "enable_pairing", sunshine.enable_pairing);
    bool_f(vars, "enable_discovery", sunshine.enable_discovery);
    bool_f(vars, "forward_rumble", input.forward_rumble);

    int port = sunshine.port;
    int_between_f(vars, "port"s, port, {1024 + nvhttp::PORT_HTTPS, 65535 - rtsp_stream::RTSP_SETUP_PORT});
    sunshine.port = (std::uint16_t) port;

    string_restricted_f(vars, "address_family", sunshine.address_family, {"ipv4"sv, "both"sv});
    {
      std::string bind_address;
      string_f(vars, "bind_address", bind_address);
      const auto address_family = net::af_from_enum_string(sunshine.address_family);
      if (net::is_valid_bind_address(bind_address, address_family)) {
        sunshine.bind_address = std::move(bind_address);
      } else {
        BOOST_LOG(error) << "Invalid bind_address ["sv << bind_address
                         << "]: it is not a valid "sv
                         << (address_family == net::IPV4 ? "IPv4"sv : "IPv6"sv)
                         << " address for address_family="sv << sunshine.address_family
                         << "; network startup will fail closed instead of listening on all interfaces."sv;
        // Preserve an explicit invalid restriction. Every listener will reject it rather than
        // translating a typo into the empty/wildcard setting and unexpectedly widening exposure.
        sunshine.bind_address = std::move(bind_address);
      }
    }

    string_restricted_f(vars, "locale", config::sunshine.locale, {
                                                                   "bg"sv,  // Bulgarian
                                                                   "cs"sv,  // Czech
                                                                   "de"sv,  // German
                                                                   "en"sv,  // English
                                                                   "en_GB"sv,  // English (UK)
                                                                   "en_US"sv,  // English (US)
                                                                   "es"sv,  // Spanish
                                                                   "fr"sv,  // French
                                                                   "hu"sv,  // Hungarian
                                                                   "it"sv,  // Italian
                                                                   "ja"sv,  // Japanese
                                                                   "ko"sv,  // Korean
                                                                   "pl"sv,  // Polish
                                                                   "pt"sv,  // Portuguese
                                                                   "pt_BR"sv,  // Portuguese (Brazilian)
                                                                   "ru"sv,  // Russian
                                                                   "sv"sv,  // Swedish
                                                                   "tr"sv,  // Turkish
                                                                   "uk"sv,  // Ukrainian
                                                                   "vi"sv,  // Vietnamese
                                                                   "zh"sv,  // Chinese
                                                                   "zh_TW"sv,  // Chinese (Traditional)
                                                                 });

    std::string log_level_string;
    string_f(vars, "min_log_level", log_level_string);

    if (!log_level_string.empty()) {
      if (log_level_string == "verbose"sv) {
        sunshine.min_log_level = 0;
      } else if (log_level_string == "debug"sv) {
        sunshine.min_log_level = 1;
      } else if (log_level_string == "info"sv) {
        sunshine.min_log_level = 2;
      } else if (log_level_string == "warning"sv) {
        sunshine.min_log_level = 3;
      } else if (log_level_string == "error"sv) {
        sunshine.min_log_level = 4;
      } else if (log_level_string == "fatal"sv) {
        sunshine.min_log_level = 5;
      } else if (log_level_string == "none"sv) {
        sunshine.min_log_level = 6;
      } else {
        // accept digit directly
        auto val = log_level_string[0];
        if (val >= '0' && val < '7') {
          sunshine.min_log_level = val - '0';
        }
      }
    }

    bool_f(vars, "diagnostics", sunshine.diagnostics_enabled);

    auto it = vars.find("flags"s);
    if (it != std::end(vars)) {
      apply_flags(it->second.c_str());

      vars.erase(it);
    }

    // Managed live by the Windows local AR-display controller and preserved here so it does not
    // appear as an unrecognized user-facing option during normal configuration parsing.
    vars.erase("ar_glass_devices");

    if (sunshine.min_log_level <= 3) {
      for (auto &[var, _] : vars) {
        std::cout << "Warning: Unrecognized configurable option ["sv << var << ']' << std::endl;
      }
    }
  }

  int parse(int argc, char *argv[]) {
    std::unordered_map<std::string, std::string> cmd_vars;
#ifdef _WIN32
    bool shortcut_launch = false;
    bool service_admin_launch = false;
#endif

    for (auto x = 1; x < argc; ++x) {
      auto line = argv[x];

      if (line == "--help"sv) {
        logging::print_help(*argv);
        return 1;
      }
#ifdef _WIN32
      else if (line == "--shortcut"sv) {
        shortcut_launch = true;
      } else if (line == "--shortcut-admin"sv) {
        service_admin_launch = true;
      }
#endif
      else if (*line == '-') {
        if (*(line + 1) == '-') {
          sunshine.cmd.name = line + 2;
          sunshine.cmd.argc = argc - x - 1;
          sunshine.cmd.argv = argv + x + 1;

          break;
        }
        if (apply_flags(line + 1)) {
          logging::print_help(*argv);
          return -1;
        }
      } else {
        auto line_end = line + strlen(line);

        auto pos = std::find(line, line_end, '=');
        if (pos == line_end) {
          sunshine.config_file = line;
        } else {
          TUPLE_EL(var, 1, parse_option(line, line_end));
          if (!var) {
            logging::print_help(*argv);
            return -1;
          }

          TUPLE_EL_REF(name, 0, *var);

          auto it = cmd_vars.find(name);
          if (it != std::end(cmd_vars)) {
            cmd_vars.erase(it);
          }

          cmd_vars.emplace(std::move(*var));
        }
      }
    }

    bool config_loaded = false;
    try {
      // Create appdata folder if it does not exist
      file_handler::make_directory(platf::appdata().string());

      // Create an empty config file if it does not exist.
      if (!fs::exists(sunshine.config_file)) {
        std::ofstream {sunshine.config_file};
      }

      // Read config file
      auto vars = parse_config(file_handler::read_file(sunshine.config_file.c_str()));

      for (auto &[name, value] : cmd_vars) {
        vars.insert_or_assign(std::move(name), std::move(value));
      }

      // Apply the config. Note: This will try to create any paths
      // referenced in the config, so we may receive exceptions if
      // the path is incorrect or inaccessible.
      apply_config(std::move(vars));
      config_loaded = true;
    } catch (const std::filesystem::filesystem_error &err) {
      BOOST_LOG(fatal) << "Failed to apply config: "sv << err.what();
    } catch (const boost::filesystem::filesystem_error &err) {
      BOOST_LOG(fatal) << "Failed to apply config: "sv << err.what();
    }

#ifdef _WIN32
    // UCRT64 raises an access denied exception if launching from the shortcut
    // as non-admin and the config folder is not yet present; we can defer
    // so that service instance will do the work instead.

    if (!config_loaded && !shortcut_launch) {
      BOOST_LOG(fatal) << "To relaunch Apollo successfully, use the shortcut in the Start Menu. Do not run sunshine.exe manually."sv;
      std::this_thread::sleep_for(10s);
#else
    if (!config_loaded) {
#endif
      return -1;
    }

#ifdef _WIN32
    // We have to wait until the config is loaded to handle these launches,
    // because we need to have the correct base port loaded in our config.
    // Exception: UCRT64 shortcut_launch instances may have no config loaded due to
    // insufficient permissions to create folder; port defaults will be acceptable.
    if (service_admin_launch) {
      // This is a relaunch as admin to start the service
      service_ctrl::start_service();

      // Always return 1 to ensure Sunshine doesn't start normally
      return 1;
    }
    if (shortcut_launch) {
      if (!service_ctrl::is_service_running()) {
        // If the service isn't running, relaunch ourselves as admin to start it
        WCHAR executable[MAX_PATH];
        GetModuleFileNameW(nullptr, executable, ARRAYSIZE(executable));

        SHELLEXECUTEINFOW shell_exec_info {};
        shell_exec_info.cbSize = sizeof(shell_exec_info);
        shell_exec_info.fMask = SEE_MASK_NOASYNC | SEE_MASK_NO_CONSOLE | SEE_MASK_NOCLOSEPROCESS;
        shell_exec_info.lpVerb = L"runas";
        shell_exec_info.lpFile = executable;
        shell_exec_info.lpParameters = L"--shortcut-admin";
        shell_exec_info.nShow = SW_NORMAL;
        if (!ShellExecuteExW(&shell_exec_info)) {
          auto winerr = GetLastError();
          BOOST_LOG(error) << "Failed executing shell command: " << winerr << std::endl;
          return 1;
        }

        // Wait for the elevated process to finish starting the service
        WaitForSingleObject(shell_exec_info.hProcess, INFINITE);
        CloseHandle(shell_exec_info.hProcess);

        // Wait for the UI to be ready for connections
        service_ctrl::wait_for_ui_ready();
      }

      // Launch the web UI
      launch_ui();

      // Always return 1 to ensure Sunshine doesn't start normally
      return 1;
    }
#endif

    return 0;
  }
}  // namespace config

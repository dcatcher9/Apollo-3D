/**
 * @file src/config.h
 * @brief Declarations for the configuration of Sunshine.
 */
#pragma once

// standard includes
#include <bitset>
#include <chrono>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// local includes
#include "nvenc/nvenc_config.h"

namespace config {
  // Authenticated Host SBS V2 policy: the fixed inference-shape budget and the private
  // normalized-depth analysis used by scene cuts. This calibration is not configurable; every
  // depth estimator (live, offline conversion, and the evaluation harness) uses these values.
  namespace host_sbs_v2_live_calibration {
    inline constexpr double depth_ema = 0.5;
    inline constexpr double edge_change = 0.05;
    inline constexpr double edge_gradient = 0.02;
    inline constexpr double edge_strength = 0.25;
    inline constexpr double minmax_ema = 0.18;
    inline constexpr int depth_short_side = 432;
    inline constexpr double depth_max_aspect = 4.0;
  }

  // track modified config options
  inline std::unordered_map<std::string, std::string> modified_config_settings;

  // A DA-V2 model identity. Host SBS resolves its sole authenticated production entry directly
  // from the generated Depth Coordinate V2 calibration contract.
  struct depth_model_info {
    std::string name;  ///< ONNX file stem and logical model key; TensorRT engines add a build-recipe suffix.
    std::string url;  ///< Download source for <name>.onnx if absent. Empty = local-only.
  };

  struct video_t {
    nvenc::nvenc_config nv;
    bool nv_realtime_hags;
    bool nv_opengl_vulkan_on_dxgi;
    bool nv_sunshine_high_power_mode;

    std::string adapter_name;
    std::string output_name;

    int max_bitrate;  // Maximum bitrate, sets ceiling in kbps for bitrate requested from client
    double minimum_fps_target;  ///< Lowest framerate that will be used when streaming. Range 0-1000, 0 = half of client's requested framerate.

    // Host/offline 2D->3D side-by-side (SBS) settings. Every geometry-producing path runs the
    // Depth Coordinate V2 pipeline with its fixed calibration; only the explicit shared controls
    // below remain configurable.
    struct sbs_t {
      double pop_strength = 1.75;  ///< Literal live V2 stereo strength (0.25-2) and offline conversion base strength.
      int max_encode_width = 8192;  ///< Configured packed Host SBS width cap. Output also respects both runtime NVENC axes (RTX 5080: H.264 4096x4096, HEVC/AV1 8192x8192) with aspect-preserving scaling.
      bool cuda_graph = true;  ///< Capture/replay the TensorRT enqueue when mapped D3D buffer addresses remain stable. Falls back to ordinary enqueue when unsupported.
    };

    sbs_t sbs;
  };

  struct audio_t {
    std::string sink;
    std::string virtual_sink;
  };

  struct stream_t {
    std::chrono::milliseconds ping_timeout;

    // Retain the launched app, virtual display, and process-wide streaming setup after the
    // remote client disconnects so a transient reconnect can resume without rebuilding state.
    std::chrono::milliseconds session_resume_grace;

    std::string file_apps;

    int fec_percentage;

    // Optional ceiling for the client-requested video packet size. Zero disables the ceiling.
    int packet_size_limit;
  };

  struct nvhttp_t {
    // Could be any of the following values:
    // pc|lan|wan
    std::string origin_web_ui_allowed;

    std::string pkey;
    std::string cert;

    std::string sunshine_name;

    std::string file_state;
  };

  struct input_t {
    std::unordered_map<int, int> keybindings;

    std::chrono::milliseconds back_button_timeout;
    std::chrono::milliseconds key_repeat_delay;
    std::chrono::duration<double> key_repeat_period;

    std::string gamepad;
    bool ds4_back_as_touchpad_click;
    bool motion_as_ds4;
    bool touchpad_as_ds4;

    bool always_send_scancodes;

    bool high_resolution_scrolling;
    bool native_pen_touch;

    bool forward_rumble;
  };

  namespace flag {
    enum flag_e : std::size_t {
      FRESH_STATE = 0,  ///< Do not load or save state
      FLAG_SIZE  ///< Number of flags
    };
  }  // namespace flag

  struct prep_cmd_t {
    prep_cmd_t(std::string &&do_cmd, std::string &&undo_cmd, bool &&elevated):
        do_cmd(std::move(do_cmd)),
        undo_cmd(std::move(undo_cmd)),
        elevated(std::move(elevated)) {
    }

    explicit prep_cmd_t(std::string &&do_cmd, bool &&elevated):
        do_cmd(std::move(do_cmd)),
        elevated(std::move(elevated)) {
    }

    std::string do_cmd;
    std::string undo_cmd;
    bool elevated;
  };

  inline constexpr bool default_virtual_display_restart_explorer = false;

  struct sunshine_t {
    bool hide_tray_controls;
    bool enable_pairing;
    bool enable_discovery;
    bool virtual_display_restart_explorer;
    std::string locale;
    int min_log_level;
    bool diagnostics_enabled;
    std::bitset<flag::FLAG_SIZE> flags;
    std::string credentials_file;

    std::string username;
    std::string password;
    std::string salt;

    std::string config_file;

    struct cmd_t {
      std::string name;
      int argc;
      char **argv;
    } cmd;

    std::uint16_t port;
    std::string address_family;
    std::string bind_address;

    std::string log_file;
    bool system_tray;
    std::vector<prep_cmd_t> prep_cmds;
  };

  extern video_t video;
  extern audio_t audio;
  extern stream_t stream;
  extern nvhttp_t nvhttp;
  extern input_t input;
  extern sunshine_t sunshine;

  int parse(int argc, char *argv[]);
  std::unordered_map<std::string, std::string> parse_config(const std::string_view &file_content);
}  // namespace config

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
  // track modified config options
  inline std::unordered_map<std::string, std::string> modified_config_settings;

  // A configured DA-V2 model for the SBS 3D pipeline. The startup profile owns model selection; this
  // registry resolves names into local file stems and optional download sources.
  struct depth_model_info {
    std::string name;  ///< ONNX file stem and logical model key; TensorRT engines add a build-recipe suffix.
    std::string url;  ///< Download source for <name>.onnx if absent. Empty = local-only.
  };

  /// Built-in depth-model roster.
  const std::vector<depth_model_info> &depth_model_registry();

  struct video_t {
    nvenc::nvenc_config nv;
    bool nv_realtime_hags;
    bool nv_opengl_vulkan_on_dxgi;
    bool nv_sunshine_high_power_mode;

    std::string adapter_name;
    std::string output_name;

    int max_bitrate;  // Maximum bitrate, sets ceiling in kbps for bitrate requested from client
    double minimum_fps_target;  ///< Lowest framerate that will be used when streaming. Range 0-1000, 0 = half of client's requested framerate.

    // Real-time 2D->3D side-by-side (SBS) depth reprojection tuning.
    // The selected host profile is resolved once at startup, then explicit sbs_3d_* overrides
    // are applied on top. Profiles are configuration presets only; they are not switched live.
    struct sbs_t {
      std::string profile = "apollo";  ///< Startup quality preset. Custom names use sbs_3d_profile_<name>_<parameter> keys.
      double pop_strength = 1.25;  ///< Production stereo-parallax multiplier (0.25-2). Literal reference runs bypass production scaling in the offline harness.
      bool adaptive_pop = true;  ///< Select pop once per scene from depth-edge risk, then hold it constant until a hard cut.
      double adaptive_pop_max = 1.30;  ///< Validated absolute ceiling for adaptive pop (pop_strength-2). Values below pop_strength clamp to the floor.
      double ema = 0.5;  ///< Temporal smoothing blend for the depth map (0-1). Higher = snappier, lower = more stable.
      double ema_edge_change = 0.05;  ///< Edge-selective EMA: minimum current-vs-history depth change. 0 disables it.
      double ema_edge_gradient = 0.02;  ///< Edge-selective EMA: minimum current depth gradient.
      double ema_edge_strength = 0.25;  ///< Edge-selective EMA blend toward current depth inside the mask.
      int depth_short_side = 432;  ///< Depth map short-side resolution, clamped to the frame's native short side. At 16:9 this maps to about 768x432, matching the VisionDepth3D reference input.
      double depth_max_aspect = 4.0;  ///< Aspect-ratio cap (long side <= short * this). Bounds worst-case inference cost on ultrawide.
      double minmax_ema = 0.18;  ///< Temporal EMA blend for the normalized disparity min/max (0-1). Lower = steadier depth scale, higher = adapts faster.
      double subject_lock = 0.5;  ///< Validated subject anchor compromise. 1 pins the subject exactly; 0 leaves the Bestv2 bands unanchored.
      double subject_recenter = 0.35;  ///< How strongly the depth field is shifted to put the tracked subject at mid-depth before the band mapping (0-1).
      bool subject_stretch = true;  ///< Bestv2 shape_depth_for_pop stretch: rescale the permanent P5/P95 band to [0,1].
      std::string zero_plane = "legacy";  ///< Shot-latched screen-plane anchor: legacy, subject, median, or background. Experimental modes preserve disparity range and eye symmetry.
      std::string depth_model = "depth_anything_v2_fp16";  ///< Local ONNX stem/logical model name. Identifies models so each gets its own recipe-specific engine cache.
      std::string depth_model_url = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx";  ///< URL to download the depth model ONNX from if <depth_model>.onnx is absent. Point this (and depth_model) elsewhere to use a different model.
      int max_encode_width = 8192;  ///< Configured maximum packed Host SBS width. The effective cap is the lower of this value and the selected codec's NVENC capability (RTX 5080: H.264 4096, HEVC/AV1 8192); wider requests are aspect-preservingly scaled.
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

  struct sunshine_t {
    bool hide_tray_controls;
    bool enable_pairing;
    bool enable_discovery;
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

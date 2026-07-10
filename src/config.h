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

  // A selectable depth model for the SBS 3D pipeline. The built-in roster
  // (depth_model_registry()) is indexed by a stable id used as the wire value of the
  // 0x3005 "Set Depth Model" control message, so the client can switch models mid-stream.
  // The per-model fields capture how each model differs from the current DA-V2-small
  // default; the ones past name/url are consumed by the depth estimator in Phase C
  // (DA-V3: rank-5 input, inverted depth, confidence output).
  struct depth_model_info {
    std::string name;  ///< File stem: <name>.onnx / <name>.engine in the assets dir (also the g_engines key).
    std::string url;  ///< Download source for <name>.onnx if absent. Empty = local-only (must be pre-staged, e.g. locally fp16-converted DA-V3).
    int input_rank = 4;  ///< ONNX pixel_values rank. 4 = [1,3,H,W] (DA-V2); 5 = [1,1,3,H,W] (DA-V3 any-view). Rank-5 engine support lands in Phase C.
    int patch = 14;  ///< Patch size; inference dims are rounded to a multiple of this.
    int output_transform = 0;  ///< Applied to raw model output before normalization. 0 = identity (DA-V2 disparity); 1 = shifted reciprocal disparity = 1/(depth + sbs.depth_shift) (DA-V3). The shift bounds the near-pixel spike (depth->0 stays finite), so no outliers hijack the min/max -- the clean fix vs the old 1/depth + robust-clip band-aids (iw3's approach).
    bool keep_confidence = false;  ///< Keep the model's confidence output (DA-V3) for confidence-guided warp (Phase C4).
    bool fixed_shape = false;  ///< The ONNX has a FIXED input resolution (no dynamic H/W) — build the engine WITHOUT an optimization profile and skip runtime setInputShape. Needed for models whose dynamic-shape export bakes resolution-dependent shape math. The runtime depth resolution MUST equal the export resolution. Superseded for DA3MONO by dynamic_width.
    bool dynamic_width = false;  ///< The ONNX has a FIXED input HEIGHT (baked short side) but DYNAMIC width — build a height-pinned, width-range optimization profile and set only the width at runtime. DA3MONO-LARGE uses this: its DINOv3 pos-embed export bakes the patch grid, but with height locked at fixed_h only the width patch count varies, which exports cleanly. Covers every landscape aspect (16:9 .. ultrawide) with ONE engine. Mutually exclusive with fixed_shape.
    int fixed_h = 0;  ///< Baked input height (patch multiple) for dynamic_width models, e.g. 336. Runtime depth height is pinned to this; only the width tracks the source aspect.
    std::string input_tensor = "pixel_values";  ///< Input tensor name bound by the pipeline.
    std::string output_tensor = "predicted_depth";  ///< Depth output tensor name bound by the pipeline.
    double depth_fps_override = 0.0;  ///< Per-model target depth fps; 0 = use sbs.depth_fps. For heavier models that can't hold the global rate.
  };

  /// Built-in depth-model roster. Index = the 0x3005 wire id. Stable ordering.
  const std::vector<depth_model_info> &depth_model_registry();

  struct video_t {
    bool headless_mode;
    bool limit_framerate;
    bool double_refreshrate;
    // ffmpeg params
    int qp;  // higher == more compression and less quality

    int hevc_mode;
    int av1_mode;

    int min_threads;  // Minimum number of threads/slices for CPU encoding

    struct {
      std::string sw_preset;
      std::string sw_tune;
      std::optional<int> svtav1_preset;
    } sw;

    nvenc::nvenc_config nv;
    bool nv_realtime_hags;
    bool nv_opengl_vulkan_on_dxgi;
    bool nv_sunshine_high_power_mode;

    struct {
      int preset;
      int multipass;
      int h264_coder;
      int aq;
      int vbv_percentage_increase;
    } nv_legacy;

    struct {
      std::optional<int> qsv_preset;
      std::optional<int> qsv_cavlc;
      bool qsv_slow_hevc;
    } qsv;

    struct {
      std::optional<int> amd_usage_h264;
      std::optional<int> amd_usage_hevc;
      std::optional<int> amd_usage_av1;
      std::optional<int> amd_rc_h264;
      std::optional<int> amd_rc_hevc;
      std::optional<int> amd_rc_av1;
      std::optional<int> amd_enforce_hrd;
      std::optional<int> amd_quality_h264;
      std::optional<int> amd_quality_hevc;
      std::optional<int> amd_quality_av1;
      std::optional<int> amd_preanalysis;
      std::optional<int> amd_vbaq;
      int amd_coder;
    } amd;

    struct {
      int vt_allow_sw;
      int vt_require_sw;
      int vt_realtime;
      int vt_coder;
    } vt;

    struct {
      bool strict_rc_buffer;
    } vaapi;

    std::string capture;
    std::string encoder;
    std::string adapter_name;
    std::string output_name;

    struct dd_t {
      struct workarounds_t {
        std::chrono::milliseconds hdr_toggle_delay;  ///< Specify whether to apply HDR high-contrast color workaround and what delay to use.
      };

      enum class config_option_e {
        disabled,  ///< Disable the configuration for the device.
        verify_only,  ///< @seealso{display_device::SingleDisplayConfiguration::DevicePreparation}
        ensure_active,  ///< @seealso{display_device::SingleDisplayConfiguration::DevicePreparation}
        ensure_primary,  ///< @seealso{display_device::SingleDisplayConfiguration::DevicePreparation}
        ensure_only_display  ///< @seealso{display_device::SingleDisplayConfiguration::DevicePreparation}
      };

      enum class resolution_option_e {
        disabled,  ///< Do not change resolution.
        automatic,  ///< Change resolution and use the one received from Moonlight.
        manual  ///< Change resolution and use the manually provided one.
      };

      enum class refresh_rate_option_e {
        disabled,  ///< Do not change refresh rate.
        automatic,  ///< Change refresh rate and use the one received from Moonlight.
        manual  ///< Change refresh rate and use the manually provided one.
      };

      enum class hdr_option_e {
        disabled,  ///< Do not change HDR settings.
        automatic  ///< Change HDR settings and use the state requested by Moonlight.
      };

      struct mode_remapping_entry_t {
        std::string requested_resolution;
        std::string requested_fps;
        std::string final_resolution;
        std::string final_refresh_rate;
      };

      struct mode_remapping_t {
        std::vector<mode_remapping_entry_t> mixed;  ///< To be used when `resolution_option` and `refresh_rate_option` is set to `automatic`.
        std::vector<mode_remapping_entry_t> resolution_only;  ///< To be use when only `resolution_option` is set to `automatic`.
        std::vector<mode_remapping_entry_t> refresh_rate_only;  ///< To be use when only `refresh_rate_option` is set to `automatic`.
      };

      config_option_e configuration_option;
      resolution_option_e resolution_option;
      std::string manual_resolution;  ///< Manual resolution in case `resolution_option == resolution_option_e::manual`.
      refresh_rate_option_e refresh_rate_option;
      std::string manual_refresh_rate;  ///< Manual refresh rate in case `refresh_rate_option == refresh_rate_option_e::manual`.
      hdr_option_e hdr_option;
      std::chrono::milliseconds config_revert_delay;  ///< Time to wait until settings are reverted (after stream ends/app exists).
      bool config_revert_on_disconnect;  ///< Specify whether to revert display configuration on client disconnect.
      mode_remapping_t mode_remapping;
      workarounds_t wa;
    } dd;

    int max_bitrate;  // Maximum bitrate, sets ceiling in kbps for bitrate requested from client
    double minimum_fps_target;  ///< Lowest framerate that will be used when streaming. Range 0-1000, 0 = half of client's requested framerate.

    std::string fallback_mode;
    bool isolated_virtual_display_option;
    bool ignore_encoder_probe_failure;

    // Real-time 2D->3D side-by-side (SBS) depth reprojection tuning.
    // Defaults live HERE (member initializers) -- the single source of truth. config.cpp
    // only parses the sbs_3d_* overrides; do not re-list defaults there.
    struct sbs_t {
      std::string warp = "apollo";  ///< Geometry implementation: "apollo" = occlusion-aware backward probe; "vd3d" = Bestv2 backward/forward hybrid.
      std::string shift_profile = "apollo";  ///< Disparity field: "apollo" = normalized divergence/focal mapping; "bestv2" = VisionDepth3D Bestv2's pixel-calibrated FG/MG/BG bands, subject anchor, zero-parallax trim, dynamic convergence, and 7.1% safety cap.
      double divergence = 0.0135;  ///< Parallax gain: signed parallax = (depth - focal_plane) * divergence, as a fraction of image width. 0 = flat (zero-parallax passthrough).
      double focal_plane = 0.5;  ///< Zero-parallax plane in normalized depth [0,1]; lower pushes more of the scene forward.
      double ema = 0.6;  ///< Temporal smoothing blend for the depth map (0-1). Higher = snappier, lower = more stable.
      bool ema_pixel_first = false;  ///< Per-pixel temporal EMA order. false (default) = range->pixel (normalize the raw disparity with the EMA'd bounds, THEN per-pixel EMA the normalized depth). true = pixel->range (VD3D order: per-pixel EMA the raw disparity FIRST, then normalize) -- smooths in the model's native units so a drifting normalization range re-maps a self-consistent history each frame. A/B lever.
      int depth_short_side = 432;  ///< Depth map short-side resolution, clamped to the frame's native short side. At 16:9 this maps to about 768x432, matching the VisionDepth3D reference input.
      double depth_max_aspect = 4.0;  ///< Aspect-ratio cap (long side <= short * this). Bounds worst-case inference cost on ultrawide.
      double minmax_ema = 0.1;  ///< Temporal EMA blend for the normalized disparity min/max (0-1). Lower = steadier depth scale, higher = adapts faster.
      double norm_pct_lo = 2.0;  ///< Robust normalization low percentile (0-50). 0 uses the raw minimum. The raw min/max reduction still defines the histogram range.
      double norm_pct_hi = 98.0;  ///< Robust normalization high percentile (50-100). 100 uses the raw maximum. Defaults match VisionDepth3D's p2/p98 render-stage normalization.
      bool subject_track = false;  ///< VD3D-style shaped disparity: estimate the tracked subject's depth (center/smoothness-weighted percentile, EMA'd), recenter depth around it, map depth->parallax through near/mid/far Gaussian bands, and anchor the subject at the screen plane by subtracting subject_lock x the subject's own parallax. Replaces the linear (depth-focal)*divergence mapping; divergence stays the master gain. Also cancels global depth-scale drift at the subject.
      double subject_lock = 0.95;  ///< Fraction of the tracked subject's parallax subtracted everywhere (0-1). ~1 = the subject sits exactly at the screen plane (VD3D Bestv2: 0.95); 0 = no anchoring (bands only).
      double subject_recenter = 0.35;  ///< How strongly the depth field is shifted to put the tracked subject at mid-depth before the band mapping (0-1, VD3D recenter_strength).
      bool subject_stretch = true;  ///< VD3D shape_depth_for_pop stretch: rescale the [stretch_lo, stretch_hi] percentile band to [0,1] before band mapping. Requires subject_track.
      double stretch_lo = 0.05;  ///< Low percentile for subject_stretch (VD3D depth_stretch_lo, Bestv2 0.05).
      double stretch_hi = 0.95;  ///< High percentile for subject_stretch (VD3D depth_stretch_hi, Bestv2 0.95).
      double subject_plane_lock = 0.0;  ///< VD3D apply_subject_plane_lock: additionally flatten residual disparity WITHIN the subject depth band (local), on top of the global subject_lock. 0 = off; VD3D Bestv2 0.28. Requires subject_track.
      double subject_plane_width = 0.12;  ///< Half-width (in normalized depth) of the subject band for subject_plane_lock (VD3D subject_plane_lock_width, Bestv2 0.12).
      double foreground_curvature = 0.0;  ///< VD3D foreground-volume shaping. Pushes near-region interiors toward the viewer with a centered elliptical profile. 0 = off; VD3D Bestv2 0.07.
      double vd3d_forward_blend = 0.65;  ///< VD3D hybrid weight: 0 = classic backward grid warp, 1 = depth-ordered forward splat. Bestv2 code uses 0.65.
      int vd3d_fill_radius = 96;  ///< Maximum horizontal forward-splat hole-fill distance in pixels. Bestv2 code uses 96.
      double minmax_snap = 1.6;  ///< Scene-cut snap: when a frame's raw depth range (or its center) jumps by more than this factor vs the EMA'd range, snap the normalization scale to the new scene instead of slowly blending (which makes depth "swim" for ~0.2-0.7s after a hard cut). 0 = off (always blend).
      double range_floor = 0.0;  ///< Range floor (0 = off): when the current depth range drops below this fraction of a slow-max reference range (near-flat content, e.g. a desktop page), compress the depth contrast toward the focal plane so min/max normalization doesn't stretch it to full parallax and amplify the model's hallucinated flat-scene structure. ~0.5 to enable; experimental.
      double depth_fps = 45.0;  ///< Target depth-update rate. Inference interval is auto-derived from the measured video fps (interval = round(video_fps / depth_fps)). 0 = update every frame.
      int parallax_steps = 24;  ///< Horizontal probes per eye in the SBS reprojection (runs full-res every frame). REQUIRED >= 22 with guided depth: probe spacing must stay below the smoothed depth transition (~8px) or the crossing search dithers at silhouettes.
      double border_fade = 0.02;  ///< Ramp parallax to zero within this fraction of the left/right frame edges to avoid stereo "window violations". 0 = off; ~0.02-0.05 typical.
      std::string depth_model = "depth_anything_v2_fp16";  ///< Local name/stem for the depth model files (<name>.onnx / <name>.engine). Identifies the model so different models coexist, each with its own cached engine.
      std::string depth_model_url = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx";  ///< URL to download the depth model ONNX from if <depth_model>.onnx is absent. Point this (and depth_model) elsewhere to use a different model.
      double depth_shift = 0.2;  ///< Shift in the DA-V3 disparity transform 1/(depth + depth_shift) (models with output_transform=1). Bounds the near spike; also the foreground-scale/pop knob (smaller = more pop). iw3 default 0.2. Ignored by DA-V2 (output_transform=0).
      std::string prebuild_models = "";  ///< Comma-separated depth-model names (registry stems, e.g. "depth_anything_v3_small_fp16,depth_anything_v3_base_fp16") to build TensorRT engines for AT STARTUP, in addition to the active model. Makes a mid-stream switch to them instant instead of a first-use build (which streams flat while building). Empty = only the active model.
      int max_encode_width = 8192;  ///< Max encoder output width for host SBS. SBS doubles the client width to 2W; if 2W exceeds this, the host caps the packed frame to this width (scaling height to keep the per-eye aspect) rather than failing NVENC create. NVENC HEVC/AV1 = 8192, H.264 = 4096.
      double depth_floor = 0.25;  ///< Far-depth compression in the reprojection (d' = floor + (1-floor)*d). Narrows the disocclusion band at foreground silhouettes (its width scales with the near-far parallax gap). 0 = off; ~0.2-0.4 typical.
      bool guided_upsample = false;  ///< Apollo-only color-guided depth upsample. Snaps soft depth silhouettes toward color edges at 2x depth resolution. Off in the VD3D-parity configuration.
      double guided_sigma = 0.1;  ///< Color-distance sigma for the guided upsample (tonemapped RGB, 0-1 domain). Lower = stricter edge snapping (risk: speckle on textured surfaces), higher = smoother.
      double movie_depth_fps = 0.0;  ///< Depth-update rate override for MOVIE mode (its model is the heavy DA3MONO and film is slow content, so a lower rate is cheap and invisible). 0 = use depth_fps.
      bool perf_stats = false;  ///< Emit per-stage host-SBS timing (depth inference + convert CPU) as a rolling p50/p95/max log line + sbs_perf.json snapshot. Off by default (the perf benchmark; see docs/sbs-benchmark-plan.md).
    } sbs;
  };

  struct audio_t {
    std::string sink;
    std::string virtual_sink;
    bool stream;
    bool install_steam_drivers;
    bool keep_default;
    bool auto_capture;
  };

  constexpr int ENCRYPTION_MODE_NEVER = 0;  // Never use video encryption, even if the client supports it
  constexpr int ENCRYPTION_MODE_OPPORTUNISTIC = 1;  // Use video encryption if available, but stream without it if not supported
  constexpr int ENCRYPTION_MODE_MANDATORY = 2;  // Always use video encryption and refuse clients that can't encrypt

  struct stream_t {
    std::chrono::milliseconds ping_timeout;

    std::string file_apps;

    int fec_percentage;

    // Video encryption settings for LAN and WAN streams
    int lan_encryption_mode;
    int wan_encryption_mode;
  };

  struct nvhttp_t {
    // Could be any of the following values:
    // pc|lan|wan
    std::string origin_web_ui_allowed;

    std::string pkey;
    std::string cert;

    std::string sunshine_name;

    std::string file_state;

    std::string external_ip;
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
    bool ds5_inputtino_randomize_mac;

    bool keyboard;
    bool mouse;
    bool controller;

    bool always_send_scancodes;

    bool high_resolution_scrolling;
    bool native_pen_touch;

    bool enable_input_only_mode;
    bool forward_rumble;
  };

  namespace flag {
    enum flag_e : std::size_t {
      PIN_STDIN = 0,  ///< Read PIN from stdin instead of http
      FRESH_STATE,  ///< Do not load or save state
      FORCE_VIDEO_HEADER_REPLACE,  ///< force replacing headers inside video data
      UPNP,  ///< Try Universal Plug 'n Play
      CONST_PIN,  ///< Use "universal" pin
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

  struct server_cmd_t {
    server_cmd_t(std::string &&cmd_name, std::string &&cmd_val, bool &&elevated):
        cmd_name(std::move(cmd_name)),
        cmd_val(std::move(cmd_val)),
        elevated(std::move(elevated)) {
    }
    std::string cmd_name;
    std::string cmd_val;
    bool elevated;
  };

  struct sunshine_t {
    bool hide_tray_controls;
    bool enable_pairing;
    bool enable_discovery;
    bool envvar_compatibility_mode;
    std::string locale;
    int min_log_level;
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

    std::string log_file;
    bool notify_pre_releases;
    bool legacy_ordering;
    bool system_tray;
    std::vector<prep_cmd_t> prep_cmds;
    std::vector<prep_cmd_t> state_cmds;
    std::vector<server_cmd_t> server_cmds;
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

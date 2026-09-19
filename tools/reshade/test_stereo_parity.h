// SPDX-License-Identifier: GPL-3.0-only
// Test-only matched input collection. Does not implement either stereo warp.
#pragma once

#include <iomanip>
#include <sstream>
#include "test_depth3d_effect.h"
#include "test_matched_disparity.h"

namespace sunshine_parity {
  namespace fs = std::filesystem;
  namespace api = reshade::api;
  inline ID3D12QueryHeap *queries = nullptr;
  inline ID3D12Resource *query_readback = nullptr;
  inline bool armed = false, started = false, finished = false, invalid = false;
  inline fs::path output_directory;

  inline void need(bool value, const char *message) {
    if (!value) {
      throw std::runtime_error(message);
    }
  }

  inline bool requested() {
    const char *value = std::getenv("SUNSHINE_STEREO_PARITY");
    need(!value || !*value || std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0, "SUNSHINE_STEREO_PARITY must be 0 or 1");
    return value && std::strcmp(value, "1") == 0;
  }

  inline unsigned timing_count(const char *name, unsigned fallback, unsigned minimum, unsigned maximum) {
    const char *text = std::getenv(name);
    if (!text || !*text) return fallback;
    unsigned value = 0;
    for (const char *digit = text; *digit; ++digit) {
      if (*digit < '0' || *digit > '9' || value > maximum / 10)
        throw std::runtime_error(std::string(name) + " must be an unsigned integer in [" +
          std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
      value = value * 10 + unsigned(*digit - '0');
      if (value > maximum)
        throw std::runtime_error(std::string(name) + " exceeds " + std::to_string(maximum));
    }
    if (value < minimum)
      throw std::runtime_error(std::string(name) + " must be at least " + std::to_string(minimum));
    return value;
  }

  inline bool sole_host_warp(api::effect_runtime *runtime, const char *effect) {
    if (std::strcmp(effect, "SunshineGame3D.fx")) return false;
    bool marked = false;
    runtime->enumerate_uniform_variables(effect, [&](api::effect_runtime *owner, api::effect_uniform_variable uniform) {
      char name[256] {};
      owner->get_uniform_variable_name(uniform, name);
      if (std::strcmp(name, "Depth_Adjustment") && std::strcmp(name, "SunshineGame3D::Depth_Adjustment")) return;
      int version = 0;
      if (owner->get_annotation_int_from_uniform_variable(uniform, "sunshine_warp_version", &version, 1)) {
        need(version == 2, "Unsupported Sunshine warp capability version");
        need(!marked, "Duplicate Sunshine warp capability owner");
        marked = true;
      }
    });
    return marked;
  }

  inline bool retired_ray_control(const char *name) {
    for (const auto retired : {"Compatibility_Power", "Performance_Level", "View_Mode", "View_Mode_Warping",
           "Warping_Masking", "Weapon_Near_Halo_Reduction", "Reconstruction_Size", "De_Artifacting", "Extended_Smoothing"})
      if (!std::strcmp(name, retired)) return true;
    return false;
  }

  template<class Fixture>
  auto capture_preparation_textures(Fixture &f, const fs::path &base, int)
    -> decltype(f.capture_parity_preparation(base), void()) {
    f.capture_parity_preparation(base);
  }

  template<class Fixture>
  void capture_preparation_textures(Fixture &, const fs::path &, long) {
    need(false, "This fixture does not support preparation capture");
  }

  template<class Fixture>
  auto capture_host_fields(Fixture &f, const fs::path &base, int)
    -> decltype(f.capture_host_warp(base), void()) { f.capture_host_warp(base); }

  template<class Fixture>
  void capture_host_fields(Fixture &, const fs::path &, long) {
    need(false, "This fixture does not support Host warp field capture");
  }

  struct timeline_uniform {
    api::effect_uniform_variable handle {};
    api::format type {};
    unsigned kind = 0; // 0=frametime, 1=timer, 2=framecount.
  };

  // This clock controls shader inputs in an explicit renderer-only experiment.
  // It does not change production clocks, camera/depth values, or adaptation.
  inline bool deterministic = false, timeline_running = false, timeline_pending = false;
  inline bool timeline_main_seen = false;
  inline api::effect_technique timeline_main {};
  inline std::vector<timeline_uniform> timeline_uniforms;
  inline std::uint64_t timeline_tick = 0;
  inline std::ostringstream timeline_frames, timeline_captures;
  constexpr float timeline_frame_ms = 1000.0f / 60.0f;

  inline void configure_timeline() {
    const char *value = std::getenv("SUNSHINE_STEREO_PARITY_DETERMINISTIC");
    need(!value || !*value || !std::strcmp(value, "0") || !std::strcmp(value, "1"),
      "SUNSHINE_STEREO_PARITY_DETERMINISTIC must be 0 or 1");
    deterministic = value && !std::strcmp(value, "1");
    need(!deterministic || requested(), "Deterministic shader time is restricted to the parity fixture");
  }

  inline void timeline_reload() {
    if (!deterministic) return;
    need(!timeline_running, "Effect reload invalidated the deterministic renderer history");
    timeline_main = {};
    timeline_main_seen = false;
    timeline_uniforms.clear();
  }

  inline void begin(api::command_list *commands) {
    if (!armed) {
      return;
    }
    if (started || !queries || !query_readback) {
      invalid = true;
      return;
    }
    reinterpret_cast<ID3D12GraphicsCommandList *>(commands->get_native())->EndQuery(queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
    started = true;
  }

  inline void end(api::command_list *commands) {
    if (!armed) {
      return;
    }
    if (!started || finished) {
      invalid = true;
      return;
    }
    auto *native = reinterpret_cast<ID3D12GraphicsCommandList *>(commands->get_native());
    native->EndQuery(queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
    native->ResolveQueryData(queries, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, query_readback, 0);
    finished = true;
  }

  inline std::uint64_t fingerprint(const void *data, size_t bytes) {
    std::uint64_t value = 14695981039346656037ull;
    const auto *input = static_cast<const std::uint8_t *>(data);
    for (size_t i = 0; i < bytes; ++i) {
      value ^= input[i];
      value *= 1099511628211ull;
    }
    return value;
  }

  inline void write_bytes(const fs::path &path, const void *data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    need(out.good(), "Cannot write parity binary artifact");
  }

  inline void persist_timeline() {
    if (!deterministic) return;
    const auto frames = timeline_frames.str(), captures = timeline_captures.str();
    write_bytes(output_directory / "timeline-frames.txt", frames.data(), frames.size());
    write_bytes(output_directory / "timeline-captures.txt", captures.data(), captures.size());
  }

  inline void record_timeline_capture(const char *label, std::uint64_t before) {
    if (!deterministic) return;
    need(timeline_running && !timeline_pending && timeline_tick > before, "Invalid deterministic capture tick");
    timeline_captures << label << " first_tick=" << before + 1 << " capture_tick=" << timeline_tick
      << " effect_steps=" << timeline_tick - before << '\n';
    persist_timeline();
  }

  inline std::vector<std::uint8_t> read_file(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    need(in.good(), "Missing parity reference file");
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
  }

  inline void manifest_file(std::ostream &out, const fs::path &path) {
    const auto bytes = read_file(path);
    out << "file=" << path.generic_string() << " bytes=" << bytes.size() << " fnv1a64=" << std::hex
        << fingerprint(bytes.data(), bytes.size()) << std::dec << '\n';
  }

  inline void provenance(const fs::path &runtime, const fs::path &shaders, const fs::path &directory) {
    if (!requested()) {
      return;
    }
    output_directory = directory / "parity";
    need(!fs::exists(output_directory), "Parity artifacts already exist; use a fresh isolated output directory");
    fs::create_directories(output_directory);
    std::ofstream out(output_directory / "provenance.txt");
    wchar_t executable[32768] {};
    need(GetModuleFileNameW(nullptr, executable, 32768) != 0, "Cannot identify parity executable");
    manifest_file(out, executable);
    manifest_file(out, runtime);
    manifest_file(out, directory / "ReShade.ini");
    manifest_file(out, directory / "preset.ini");
    for (const auto &entry : fs::recursive_directory_iterator(shaders)) {
      if (entry.is_regular_file() && (entry.path().extension() == ".fx" || entry.path().extension() == ".fxh")) {
        manifest_file(out, entry.path());
      }
    }
    out << "hash_kind=FNV1a64_noncryptographic; source/depth comparison also requires exact byte equality\n";
    need(out.good(), "Cannot write parity provenance");
  }

  inline float linear(float encoded) {
    return encoded <= .04045f ? encoded / 12.92f : std::pow((encoded + .055f) / 1.055f, 2.4f);
  }

  inline float encoded(float value) {
    return value <= .0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.f / 2.4f) - .055f;
  }

  inline float decode_half(std::uint16_t bits) {
    const float sign = bits & 0x8000 ? -1.f : 1.f;
    const unsigned exponent = (bits >> 10) & 31, fraction = bits & 1023;
    if (!exponent) {
      return sign * std::ldexp(float(fraction), -24);
    }
    if (exponent == 31) {
      return fraction ? NAN : sign * INFINITY;
    }
    return sign * std::ldexp(1.f + float(fraction) / 1024.f, int(exponent) - 15);
  }

  inline std::uint16_t encode_half(float value) {
    std::uint32_t bits {};
    std::memcpy(&bits, &value, sizeof(bits));
    const unsigned sign = (bits >> 16) & 0x8000;
    const int exponent = int((bits >> 23) & 255) - 127 + 15;
    if (exponent <= 0) {
      if (exponent < -10) {
        return std::uint16_t(sign);
      }
      return std::uint16_t(sign | (((bits & 0x7fffff) | 0x800000) + (1u << (13 - exponent))) >> (14 - exponent));
    }
    if (exponent >= 31) {
      return std::uint16_t(sign | 0x7c00);
    }
    const unsigned rounded = (unsigned(exponent) << 10) + ((bits & 0x7fffff) + 0x1000) / 0x2000;
    return std::uint16_t(sign | rounded);
  }

  struct shifts {
    double eye[3][2] {};

    double disparity(unsigned plane) const {
      return eye[plane][0] - eye[plane][1];
    }
  };

  inline std::array<float, 3> raw_planes {.02f, .005f, .0125f};

  inline void log_shifts(std::ostream &out, const char *label, const shifts &values) {
    out << std::setprecision(12);
    for (unsigned plane = 0; plane < 3; ++plane) {
      out << "landmark case=" << label << " raw=" << raw_planes[plane] << " left_px=" << values.eye[plane][0]
          << " right_px=" << values.eye[plane][1] << " binocular_px=" << values.disparity(plane)
          << " cyclopean_px=" << .5 * (values.eye[plane][0] + values.eye[plane][1]) << '\n';
    }
  }

  inline void write_reference(const fs::path &path, const shifts &value, unsigned width, unsigned height, unsigned color) {
    std::ofstream out(path);
    out << "SUNSHINE_PARITY_REFERENCE_1 " << width << ' ' << height << ' ' << color << '\n'
        << std::setprecision(17);
    for (const auto &plane : value.eye) {
      out << plane[0] << ' ' << plane[1] << '\n';
    }
    need(out.good(), "Cannot write parity calibration reference");
  }

  inline shifts read_reference(const fs::path &path, unsigned width, unsigned height, unsigned color) {
    std::ifstream in(path);
    std::string magic;
    unsigned w {}, h {}, c {};
    in >> magic >> w >> h >> c;
    need(magic == "SUNSHINE_PARITY_REFERENCE_1" && w == width && h == height && c == color, "Parity reference dimensions/color/schema mismatch");
    shifts value;
    for (auto &plane : value.eye) {
      for (double &eye : plane) {
        in >> eye;
        need(std::isfinite(eye), "Nonfinite parity reference");
      }
    }
    need(in.good(), "Invalid parity calibration reference");
    return value;
  }

  inline void uniforms(api::effect_runtime *runtime, const char *effect, const fs::path &path) {
    std::ofstream out(path);
    out << std::setprecision(10);
    runtime->enumerate_uniform_variables(effect, [&](api::effect_runtime *active, api::effect_uniform_variable variable) {
      char name[256] {}, source[128] {};
      active->get_uniform_variable_name(variable, name);
      active->get_annotation_string_from_uniform_variable(variable, "source", source);
      api::format type {};
      std::uint32_t rows {}, columns {}, arrays {};
      active->get_uniform_variable_type(variable, &type, &rows, &columns, &arrays);
      const size_t count = size_t(rows) * columns * std::max(1u, arrays);
      need(count > 0 && count <= 4096, "Unbounded parity uniform reflection");
      std::vector<float> floats(count);
      std::vector<std::uint32_t> integers(count);
      active->get_uniform_value_float(variable, floats.data(), count);
      active->get_uniform_value_uint(variable, integers.data(), count);
      out << name << " type=" << unsigned(type) << " rows=" << rows << " columns=" << columns << " array=" << arrays
          << " source=" << source << " as_float=";
      for (float v : floats) {
        out << v << ',';
      }
      out << " as_uint=";
      for (auto v : integers) {
        out << v << ',';
      }
      out << '\n';
    });
    need(out.good(), "Cannot record parity uniform snapshot");
  }

  inline void textures(api::effect_runtime *runtime, const char *effect, const fs::path &path) {
    std::ofstream out(path);
    runtime->enumerate_texture_variables(effect, [&](api::effect_runtime *active, api::effect_texture_variable variable) {
      char name[256] {};
      active->get_texture_variable_name(variable, name);
      api::resource_view view {};
      active->get_texture_binding(variable, &view, nullptr);
      if (!view.handle) {
        out << name << " unbound\n";
        return;
      }
      const auto handle = active->get_device()->get_resource_from_view(view);
      if (!handle.handle) {
        out << name << " missing_native_resource\n";
        return;
      }
      const auto desc = reinterpret_cast<ID3D12Resource *>(handle.handle)->GetDesc();
      out << name << " width=" << desc.Width << " height=" << desc.Height << " format=" << unsigned(desc.Format)
          << " mips=" << desc.MipLevels << " array=" << desc.DepthOrArraySize << " samples=" << desc.SampleDesc.Count << '\n';
    });
    need(out.good(), "Cannot record parity texture geometry");
  }

  inline bool technique_named(const char *name, const std::string &expected) {
    const std::string value(name);
    return value == expected || (value.size() > expected.size() &&
                                 value.compare(value.size() - expected.size(), expected.size(), expected) == 0 &&
                                 value[value.size() - expected.size() - 1] == ':');
  }

  inline void timeline_before_effects(api::effect_runtime *runtime, const char *effect, const char *selected) {
    if (!deterministic) return;
    if (!timeline_main.handle) {
      runtime->enumerate_techniques(effect, [&](api::effect_runtime *active, api::effect_technique technique) {
        char name[256] {};
        active->get_technique_name(technique, name);
        if (technique_named(name, selected)) {
          need(!timeline_main.handle, "Duplicate deterministic stereo technique");
          timeline_main = technique;
        }
      });
      need(timeline_main.handle != 0, "Deterministic stereo technique was not compiled");
      unsigned frame_times = 0, timers = 0;
      runtime->enumerate_uniform_variables(effect, [&](api::effect_runtime *active, api::effect_uniform_variable uniform) {
        char source[64] {};
        active->get_annotation_string_from_uniform_variable(uniform, "source", source);
        const unsigned kind = !std::strcmp(source, "frametime") ? 0 : !std::strcmp(source, "timer") ? 1 :
          !std::strcmp(source, "framecount") ? 2 : 3;
        if (kind == 3) return;
        api::format type {};
        std::uint32_t rows {}, columns {}, arrays {};
        active->get_uniform_variable_type(uniform, &type, &rows, &columns, &arrays);
        need(rows == 1 && columns == 1 && arrays == 0, "Unsupported deterministic time uniform shape");
        need(kind == 2 ? (type == api::format::r32_typeless || type == api::format::r32_uint || type == api::format::r32_sint) :
          type == api::format::r32_float, "Unsupported deterministic time uniform type");
        frame_times += kind == 0;
        timers += kind == 1;
        timeline_uniforms.push_back({uniform, type, kind});
      });
      if (sole_host_warp(runtime, effect))
        need(timeline_uniforms.empty(), "Sole Host warp unexpectedly retained shader time/history inputs");
      else
        need(frame_times == 1 && timers == 1, "Expected one original frametime and timer uniform");
    }
    if (!timeline_running) {
      // This callback precedes rendering, including the very first technique
      // execution. Allow the original information helper to initialize, but
      // never feed unknown startup depth into the stereo temporal textures.
      runtime->set_technique_state(timeline_main, false);
      timeline_main_seen = true;
      return;
    }
    need(!timeline_pending && runtime->get_technique_state(timeline_main),
      "Deterministic clock encountered a missing or overlapping stereo execution");
    need(timeline_tick < 100000, "Deterministic parity frame budget exceeded");
    const auto next = timeline_tick + 1;
    const float timer = float(double(next - 1) * (1000.0 / 60.0));
    // ReShade 6.8 updates special uniforms before this callback and uploads the
    // constant buffer afterward. Only these clock sources are overridden.
    for (const auto &uniform : timeline_uniforms) {
      if (uniform.kind < 2) {
        runtime->set_uniform_value_float(uniform.handle, uniform.kind == 0 ? timeline_frame_ms : timer);
      } else if (uniform.type == api::format::r32_typeless) {
        runtime->set_uniform_value_bool(uniform.handle, (next % 2) == 0);
      } else if (uniform.type == api::format::r32_sint) {
        runtime->set_uniform_value_int(uniform.handle, static_cast<std::int32_t>(next));
      } else {
        runtime->set_uniform_value_uint(uniform.handle, static_cast<std::uint32_t>(next));
      }
    }
    timeline_pending = true;
  }

  inline void timeline_after_technique(api::effect_runtime *runtime) {
    if (!deterministic) return;
    need(timeline_running && timeline_pending, "Stereo executed before deterministic inputs were admitted");
    const auto next = timeline_tick + 1;
    const float timer = float(double(next - 1) * (1000.0 / 60.0));
    for (const auto &uniform : timeline_uniforms) {
      if (uniform.kind < 2) {
        float actual {};
        runtime->get_uniform_value_float(uniform.handle, &actual, 1);
        need(actual == (uniform.kind == 0 ? timeline_frame_ms : timer), "Runtime replaced deterministic shader time");
      } else if (uniform.type == api::format::r32_typeless) {
        bool actual {};
        runtime->get_uniform_value_bool(uniform.handle, &actual, 1);
        need(actual == ((next % 2) == 0), "Runtime replaced deterministic frame parity");
      } else {
        std::uint32_t actual {};
        runtime->get_uniform_value_uint(uniform.handle, &actual, 1);
        need(actual == next, "Runtime replaced deterministic frame count");
      }
    }
    timeline_tick = next;
    timeline_pending = false;
    timeline_frames << std::setprecision(9) << next << ' ' << timeline_frame_ms << ' ' << timer << '\n';
  }

  inline void start_timeline(api::effect_runtime *runtime) {
    if (!deterministic) return;
    need(timeline_main_seen && timeline_main.handle && !timeline_running && !timeline_tick,
      "Deterministic history did not start from a fresh suspended technique");
    timeline_running = true;
    runtime->set_technique_state(timeline_main, true);
  }

  template<class Fixture>
  void isolate_techniques(Fixture &f, api::effect_runtime *runtime, const char *effect, bool independent, std::ostream &report) {
    const std::string selected = fs::path(effect).stem().string();
    api::effect_technique stereo {}, information {};
    unsigned owners = 0;
    runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *active, api::effect_technique technique) {
      char owner[256] {}, name[256] {};
      active->get_technique_effect_name(technique, owner);
      active->get_technique_name(technique, name);
      const bool owned = fs::path(owner).filename().string() == effect;
      const bool main = owned && technique_named(name, selected);
      // Original Information_SD initializes Info_Tex, which also supplies masks
      // and helper state to the original renderer. Preserve its natural startup
      // execution; it is not a second stereo owner and never writes the source.
      const bool helper = owned && !independent && technique_named(name, "Information_SD");
      const bool was_enabled = active->get_technique_state(technique);
      if (main) {
        ++owners;
        stereo = technique;
        active->set_technique_state(technique, !deterministic);
      } else if (helper) {
        need(!information.handle, "Duplicate original information helper");
        information = technique;
      } else {
        active->set_technique_state(technique, false);
      }
      report << "technique effect=" << owner << " name=" << name << " initial_enabled=" << was_enabled
             << " isolated_enabled=" << active->get_technique_state(technique)
             << " role=" << (main ? "stereo" : helper ? "original_initialization" :
                                                        "disabled")
             << '\n';
    });
    need(owners == 1 && runtime->get_technique_state(stereo) == !deterministic, "Parity requires exactly one selected stereo owner");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    unsigned startup_frames = 0;
    while (information.handle && runtime->get_technique_state(information) && std::chrono::steady_clock::now() < deadline) {
      f.step();
      ++startup_frames;
    }
    need(!information.handle || !runtime->get_technique_state(information), "Original initialization helper did not finish its natural timeout");
    report << "initialization_helper=finished frames_waited=" << startup_frames << '\n';
    unsigned enabled = 0;
    runtime->enumerate_techniques(nullptr, [&](api::effect_runtime *active, api::effect_technique technique) {
      enabled += active->get_technique_state(technique) ? 1u : 0u;
    });
    need(enabled == (deterministic ? 0u : 1u) && runtime->get_technique_state(stereo) == !deterministic,
      "Unexpected technique active after parity isolation");
    report << "stereo_history_during_initialization=" << (deterministic ? "not_started" : "natural") << '\n';
  }

  template<class Fixture>
  void run(Fixture &f, unsigned width, unsigned height, bool independent, api::effect_runtime *runtime, bool &ready, bool &capture, const char *original_effect = "SuperDepth3D.fx") {
    need(requested() && !output_directory.empty(), "Parity initialization missing");
    need(f.color == 1 || f.color == 2, "Initial parity harness supports SDR/scRGB only; PQ needs its separate color oracle");
    const char *effect = independent ? "SunshineDepth3D.fx" : original_effect;
    const char *reference_env = std::getenv("SUNSHINE_STEREO_PARITY_REFERENCE");
    const bool comparison = reference_env && *reference_env;
    const bool match_automatic = sunshine_matched_disparity::requested();
    const auto fixed_camera = sunshine_matched_disparity::fixed_camera();
    raw_planes = sunshine_matched_disparity::scene_depths();
    need(!fixed_camera.present || match_automatic, "Fixed camera requires explicit Automatic renderer-only comparison");
    const char *automatic_env = std::getenv("SUNSHINE_GAME3D_AUTOMATIC");
    const bool automatic = automatic_env && std::strcmp(automatic_env, "1") == 0;
    const char *aa0_env = std::getenv("SUNSHINE_STEREO_PARITY_AA0_ONLY");
    need(!aa0_env || !*aa0_env || !std::strcmp(aa0_env, "0") || !std::strcmp(aa0_env, "1"),
      "SUNSHINE_STEREO_PARITY_AA0_ONLY must be 0 or 1");
    const bool aa0_only = aa0_env && !std::strcmp(aa0_env, "1");
    need(!aa0_only || !independent, "AA0-only comparison is restricted to the original-derived effect");
    const char *legacy_aa = std::getenv("SUNSHINE_GAME3D_LEGACY_FINAL_AA");
    need(!legacy_aa || !*legacy_aa || !std::strcmp(legacy_aa, "0") || !std::strcmp(legacy_aa, "1"),
      "SUNSHINE_GAME3D_LEGACY_FINAL_AA must be 0 or 1");
    const bool legacy_aa_enabled = legacy_aa && !std::strcmp(legacy_aa, "1");
    need(!legacy_aa_enabled || (!independent && !std::strcmp(effect, "SunshineGame3D.fx")),
      "Legacy final-AA opt-in is only for frozen SunshineGame3D shaders");
    need(independent || std::strcmp(effect, "SunshineGame3D.fx") || legacy_aa_enabled || aa0_only,
      "Game3D without final AA requires explicit AA0-only parity");
    const unsigned aa_count = aa0_only ? 1u : 2u;
    const char *preparation_env = std::getenv("SUNSHINE_STEREO_PARITY_CAPTURE_PREPARATION");
    need(!preparation_env || !*preparation_env || !std::strcmp(preparation_env, "0") || !std::strcmp(preparation_env, "1"),
      "SUNSHINE_STEREO_PARITY_CAPTURE_PREPARATION must be 0 or 1");
    const bool capture_preparation = preparation_env && !std::strcmp(preparation_env, "1");
    need(!capture_preparation || (deterministic && !independent && std::strcmp(effect, "SunshineGame3D.fx") == 0),
      "Preparation capture requires deterministic full SunshineGame3D parity");
    const char *preparation_motion = std::getenv("SUNSHINE_STEREO_PARITY_MOTION");
    need(!capture_preparation || (preparation_motion && !std::strcmp(preparation_motion, "1")),
      "Preparation capture requires the static and moving parity cases");
    need(!automatic || match_automatic, "Automatic parity requires explicit renderer-only matching controls");
    need(!deterministic || !independent, "Deterministic initialization currently requires the full original-derived effect");
    need(!match_automatic || ((comparison || fixed_camera.present) && automatic && !independent && std::strcmp(effect, "SunshineGame3D.fx") == 0),
      "Automatic parity requires SunshineGame3D Automatic and a reference or explicit fixed camera");
    const char *exact_env = std::getenv("SUNSHINE_STEREO_PARITY_EXACT");
    need(!exact_env || !*exact_env || std::strcmp(exact_env, "0") == 0 || std::strcmp(exact_env, "1") == 0, "SUNSHINE_STEREO_PARITY_EXACT must be 0 or 1");
    const bool exact_pixels = comparison && exact_env && std::strcmp(exact_env, "1") == 0;
    need(!match_automatic || !exact_pixels || fixed_camera.present,
      "Exact Automatic comparison requires explicit fixed camera controls, never a fitted camera");
    const char *variant = std::getenv("SUNSHINE_STEREO_PARITY_VARIANT");
    if (!variant || !*variant) {
      variant = effect;
    }
    fs::path reference_directory;
    bool fixed_camera_reference = false;
    if (independent) {
      need(comparison, "Native parity run requires SUNSHINE_STEREO_PARITY_REFERENCE=<original parity directory>");
    }
    if (comparison) {
      reference_directory = fs::absolute(reference_env);
      need(fs::exists(reference_directory / "timeline-contract.txt") == deterministic,
        "Parity reference and treatment use different shader clock contracts");
      fixed_camera_reference = fs::exists(reference_directory / "fixed-camera.txt");
      need(fs::exists(reference_directory / "preparation-contract.txt") == capture_preparation,
        "Parity reference and treatment use different preparation capture contracts");
      need(!fixed_camera_reference || (match_automatic && fixed_camera.present),
        "Fixed-camera reference requires explicit fixed Automatic comparison controls");
    }
    std::ofstream report(output_directory / "measurements.txt");
    const bool sole_host = sole_host_warp(runtime, effect);
    report << "sole_host_warp_capability=" << sole_host << '\n';
    const char *fit_only_env = std::getenv("SUNSHINE_STEREO_PARITY_FIT_ONLY");
    need(!fit_only_env || !*fit_only_env || !std::strcmp(fit_only_env, "0") || !std::strcmp(fit_only_env, "1"),
      "SUNSHINE_STEREO_PARITY_FIT_ONLY must be 0 or 1");
    const bool fit_only = fit_only_env && !std::strcmp(fit_only_env, "1");
    need(!fit_only || (sole_host && comparison && match_automatic && deterministic && !fixed_camera.present && !exact_pixels),
      "Isolated fit-only phase requires capability-v2 deterministic Automatic matching without fixed/exact controls");
    const unsigned timing_warmup = timing_count("SUNSHINE_STEREO_PARITY_TIMING_WARMUP", 4, 0, 4096);
    const unsigned timing_samples = timing_count("SUNSHINE_STEREO_PARITY_TIMING_SAMPLES", 8, 8, 4096);
    report << "timing_contract warmup=" << timing_warmup << " samples=" << timing_samples
           << " p95_method=nearest_rank scope=effects_begin_through_selected_technique_including_COLOR_capture\n";
    unsigned fringe_width = 2;
    if (const char *fringe = std::getenv("SUNSHINE_STEREO_PARITY_FRINGE_WIDTH"); fringe && *fringe) {
      size_t used = 0;
      const unsigned long parsed = std::stoul(fringe, &used);
      need(used == std::strlen(fringe) && parsed >= 1 && parsed <= 8,
        "Parity fringe width must be an integer from 1 to 8 source pixels");
      fringe_width = unsigned(parsed);
      const std::string contract = "schema=parity-source-fringe-1\nwidth=" + std::to_string(fringe_width) + "\nforeground_mix=0.5\n";
      write_bytes(output_directory / "source-fringe-contract.txt", contract.data(), contract.size());
      if (comparison)
        need(read_file(reference_directory / "source-fringe-contract.txt") == read_file(output_directory / "source-fringe-contract.txt"),
          "Parity source fringe differs from reference");
    } else if (comparison) {
      need(!fs::exists(reference_directory / "source-fringe-contract.txt"), "Reference source fringe contract was not supplied");
    }
    report << "source_mixed_color_fringe_width_px=" << fringe_width << '\n';
    if (comparison)
      need(fs::exists(reference_directory / "final-aa-contract.txt") == aa0_only,
        "Parity control and treatment selected different final-AA case sets");
    if (aa0_only) {
      const std::string contract = "schema=final-aa-parity-1\nfinal_aa=off\ncase_branches=1\n";
      write_bytes(output_directory / "final-aa-contract.txt", contract.data(), contract.size());
      if (comparison)
        need(read_file(reference_directory / "final-aa-contract.txt") == read_file(output_directory / "final-aa-contract.txt"),
          "Parity final-AA case contract changed");
    }
    report << "final_aa_case_count=" << aa_count << " legacy_game3d_final_aa=" << legacy_aa_enabled << '\n';
    report << std::setprecision(12) << "scene_depths foreground_raw=" << raw_planes[0]
           << " background_raw=" << raw_planes[1] << " middle_raw=" << raw_planes[2] << '\n';
    if (const char *scene = std::getenv("SUNSHINE_STEREO_PARITY_SCENE"); scene && *scene) {
      const auto bytes = read_file(scene);
      write_bytes(output_directory / "explicit-scene.txt", bytes.data(), bytes.size());
      if (comparison)
        need(read_file(reference_directory / "explicit-scene.txt") == bytes, "Parity scene-depth controls differ from reference");
    } else if (comparison) {
      need(!fs::exists(reference_directory / "explicit-scene.txt"), "Reference explicit scene depths were not supplied");
    }
    if (capture_preparation) {
      const std::string contract = "schema=parity-preparation-1\ntextures=texzBufferN_P,texzBufferN_L\nsubresource=0\nall_channels_and_corners=1\nstatic_and_motion_endpoints=1\n";
      write_bytes(output_directory / "preparation-contract.txt", contract.data(), contract.size());
      if (comparison)
        need(read_file(reference_directory / "preparation-contract.txt") == read_file(output_directory / "preparation-contract.txt"),
          "Parity preparation contract changed");
    }
    auto save_preparation = [&](const std::string &label) {
      const auto before_tick = timeline_tick;
      if (capture_preparation) capture_preparation_textures(f, output_directory / label, 0);
      const char *host = std::getenv("SUNSHINE_STEREO_PARITY_CAPTURE_HOST_WARP");
      need(!host || !*host || !std::strcmp(host, "0") || !std::strcmp(host, "1"), "Host warp capture must be 0 or 1");
      if (host && !std::strcmp(host, "1")) capture_host_fields(f, output_directory / label, 0);
      need(timeline_tick == before_tick, "Preparation readback advanced the shader clock");
    };
    bool all_scene_endpoints_matched = true;
    bool all_scene_pixels_identical = true;
    report << "renderer=" << effect << " variant=" << variant << " width=" << width << " height=" << height << " source_color=" << f.color
           << " source_format=" << unsigned(f.source_format) << " depth=R32_FLOAT native_extent=1,1,0,0\n"
           << "export_format=" << unsigned(f.exported->GetDesc().Format)
           << " dump=float32_little_endian_RGBA packed_width=" << width * 2 << " packed_height=" << height
           << " eye_order=left_right rgb=linear_Rec709 scRGB_unit=80_nits alpha=renderer_confidence_unmodified\n"
           << "No_per_image_normalization_or_HDR_clamp; metrics_do_not_assume_unknown_hidden_texture\n";
    report << "camera_source=" << (fixed_camera.present ? "test_only_fixed_H_t0_controls_not_production_policy" :
      (match_automatic ? "test_only_pixel_fitted_H_t0_not_production_policy" : "original_shader_controls"))
           << " production_calibration_validated=0\n";
    // Record untouched preset/default values before the comparison fixes any
    // artistic controls. This catches accidental profile/default changes that
    // the later matched-control scenes could otherwise conceal.
    uniforms(runtime, effect, output_directory / "initial-uniforms.txt");
    isolate_techniques(f, runtime, effect, independent, report);
    ready = true;
    capture = false;
    auto set_bool = [&](const char *name, bool value, bool required = true) {
      const auto handle = f.uniform(name, required);
      if (handle.handle) {
        runtime->set_uniform_value_bool(handle, value);
      }
    };
    auto optional_float = [&](const char *name, float value) {
      const auto handle = f.uniform(name, false);
      if (handle.handle) {
        runtime->set_uniform_value_float(handle, value);
      }
    };
    auto optional_int = [&](const char *name, int value) {
      const auto handle = f.uniform(name, false);
      if (handle.handle) runtime->set_uniform_value_int(handle, value);
    };
    auto set_aa = [&](bool enabled_aa) {
      if (independent) {
        set_bool("EdgeAntialias", enabled_aa);
      } else {
        f.set_int("USE_AA", enabled_aa ? 1 : 0);
      }
    };
    if (independent) {
      set_bool("DepthView", false);
      f.set_float("Sunshine_RawAnchor", .0125f);
      f.set_float("Sunshine_RawGain", 1.f / .015f);
      f.set_float("Strength", 1);
      f.set_float("ScreenPlane", 0);
    } else {
      optional_int("Depth_Map", 1); // Removed legacy depth decoding; camera/raw inputs own current Game3D.
      f.set_int("Depth_Map_View", 0);
      f.set_float("Depth_Adjustment", 12);
      optional_float("Zero_Parallax_Distance", .18f); // Removed legacy convergence control.
      f.set_float("Compatibility_Power", 1);
      f.set_int("Performance_Level", 2);  // Highest original sample quality; preparation resolution remains source-defined.
      if (sunshine_depth3d_fixture::sharpening_available(f, effect)) f.set_float("Sharpen_Power", 0);
      optional_float("ZPD_OverShoot", 0);
      optional_float("Auto_Depth_Adjust", 0);
      set_bool("Show_Infill_Mask", false, false);
    }
    auto checked_control = [&](const char *name, api::format expected, unsigned count) {
      const auto handle = f.uniform(name); // Explicit/shared controls must exist; never silently omit one.
      api::format type {};
      std::uint32_t rows {}, columns {}, arrays {};
      runtime->get_uniform_variable_type(handle, &type, &rows, &columns, &arrays);
      need(type == expected && rows * columns == count && arrays == 0,
        "Parity control does not match the required reflected scalar/vector type");
      return handle;
    };
    auto retired_control = [&](const char *name) {
      if (!sole_host || !retired_ray_control(name)) return false;
      need(!f.uniform(name, false).handle, "Sole Host warp still exposes a retired ray control");
      report << "retired_ray_control_not_applied name=" << name << " reason=shader_capability_v2_no_consumer\n";
      return true;
    };
    if (!independent) {
      checked_control("Depth_Adjustment", api::format::r32_float, 1);
      if (sunshine_depth3d_fixture::sharpening_available(f, effect)) checked_control("Sharpen_Power", api::format::r32_float, 1);
      if (!retired_control("Compatibility_Power")) checked_control("Compatibility_Power", api::format::r32_float, 1);
      for (const auto name : {"Depth_Map_View", "View_Mode", "View_Mode_Warping", "Warping_Masking",
             "Weapon_Near_Halo_Reduction", "Performance_Level", "Reconstruction_Size"})
        if (!retired_control(name)) checked_control(name, api::format::r32_sint, 1);
      if (!retired_control("De_Artifacting")) checked_control("De_Artifacting", api::format::r32_float, 2);
      if (!retired_control("Extended_Smoothing")) checked_control("Extended_Smoothing", api::format::r32_typeless, 1);
    }
    sunshine_matched_disparity::controls(
      [&](const char *name, float value) {
        if (retired_control(name)) return;
        runtime->set_uniform_value_float(checked_control(name, api::format::r32_float, 1), value);
      },
      [&](const char *name, int value) {
        if (retired_control(name)) return;
        runtime->set_uniform_value_int(checked_control(name, api::format::r32_sint, 1), value);
      },
      [&](const char *name, const float *values, size_t count) {
        if (retired_control(name)) return;
        runtime->set_uniform_value_float(checked_control(name, api::format::r32_float, unsigned(count)), values, count);
      },
      [&](const char *name, bool value) {
        if (retired_control(name)) return;
        runtime->set_uniform_value_bool(checked_control(name, api::format::r32_typeless, 1), value);
      }, report);
    if (const char *controls = std::getenv("SUNSHINE_STEREO_PARITY_CONTROLS"); controls && *controls) {
      const auto bytes = read_file(controls);
      write_bytes(output_directory / "explicit-controls.txt", bytes.data(), bytes.size());
      if (comparison)
        need(read_file(reference_directory / "explicit-controls.txt") == bytes, "Parity artistic controls differ from reference");
    } else if (comparison) {
      need(!fs::exists(reference_directory / "explicit-controls.txt"), "Reference explicit artistic controls were not supplied");
    }
    set_aa(false);
    auto automatic_controls = [&](double gain, double zero, float reference = .05f) {
      f.set_int("Sunshine_CameraCoordinateBasis", 1);
      const float projection[2] {0, 1}, convergence[2] {reference, float(zero)};
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraProjection"), projection, 2);
      f.set_float("Sunshine_CameraDepthScale", float(gain));
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
      set_bool("Sunshine_CameraDepthReady", true);
      f.set_float("Sunshine_CameraStrengthBlend", 1);
      const float rect[4] {0, 0, 1, 1};
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraDepthRect"), rect, 4);
    };
    if (fixed_camera.present) {
      automatic_controls(fixed_camera.gain, fixed_camera.zero, fixed_camera.reference);
      const auto bytes = read_file(std::getenv("SUNSHINE_STEREO_PARITY_FIXED_CAMERA"));
      write_bytes(output_directory / "fixed-camera.txt", bytes.data(), bytes.size());
      if (fixed_camera_reference)
        need(read_file(reference_directory / "fixed-camera.txt") == bytes, "Parity fixed camera controls differ from reference");
      report << std::setprecision(17) << "fixed_camera no_refit=1 H=" << fixed_camera.gain
             << " referenceZPD=" << fixed_camera.reference << " t0=" << fixed_camera.zero
             << " effective_slope=" << double(fixed_camera.gain) * fixed_camera.reference
             << " effective_offset=" << double(fixed_camera.gain) * fixed_camera.reference * fixed_camera.zero << '\n';
    } else if (match_automatic) automatic_controls(128, .0125);

    std::vector<float> hardware(size_t(width) * height);
    const unsigned marker_top = height * 3 / 4;
    auto build_case = [&](unsigned scene, int horizontal_offset = 0) {
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const size_t pixel = size_t(y) * width + x;
          float rgb[3] {.04f, .04f, .04f};
          if (scene == 0 || y >= marker_top) {
            const unsigned top = scene == 0 ? 0 : marker_top;
            const unsigned plane = std::min(2u, (y - top) * 3 / (height - top));
            hardware[pixel] = raw_planes[plane];
            const bool marker = x >= width / 2 - 8 && x < width / 2 + 8;
            rgb[0] = rgb[1] = rgb[2] = marker ? .8f : .04f;
          } else {
            const bool mirrored = y >= marker_top / 2;
            const unsigned boundary = unsigned(int(width / 2 + ((y / 9) % 7) - 3) + horizontal_offset);
            const bool foreground = scene == 2 ?
                                      ((x % 96 == 24) || (x % 96 >= 48 && x % 96 < 50) || (x % 96 >= 72 && x % 96 < 76)) :
                                      (mirrored ? x >= boundary : x < boundary);
            hardware[pixel] = foreground ? raw_planes[0] : raw_planes[1];
            if (foreground) {
              rgb[0] = .8f;
              rgb[1] = .04f;
              rgb[2] = .04f;
            } else {
              rgb[0] = .04f;
              rgb[1] = .12f;
              rgb[2] = .65f;
            }
            if (scene == 2 && !foreground) {
              const float texture = ((x / 2 + y / 2) % 2) ? .65f : .12f;
              rgb[0] = rgb[1] = rgb[2] = texture;
            }
            if (scene == 3 && !foreground && (mirrored ? x + fringe_width >= boundary : x < boundary + fringe_width)) {
              rgb[0] = .42f;
              rgb[1] = .08f;
              rgb[2] = .345f;  // Mixed color outside the depth silhouette.
            }
            if (f.color == 2) {
              rgb[0] *= 4;
              rgb[1] -= .2f;
              rgb[2] *= 4;
            }  // Signed/highlight scRGB stays untonemapped; markers retain a fixed calibration signal.
          }
          if (f.color == 1) {
            std::uint8_t values[4] {};
            for (unsigned c = 0; c < 3; ++c) {
              values[c] = std::uint8_t(std::lround(encoded(rgb[c]) * 255));
            }
            values[3] = 255;
            std::memcpy(f.source_bytes.data() + pixel * 4, values, 4);
          } else {
            std::uint16_t values[4] {};
            for (unsigned c = 0; c < 3; ++c) {
              values[c] = encode_half(rgb[c]);
            }
            values[3] = 0x3c00;
            std::memcpy(f.source_bytes.data() + pixel * 8, values, 8);
          }
        }
      }
      f.fill_upload(f.source_upload, f.backbuffers[0]->GetDesc(), f.source_bytes.data(), f.source_footprint);
      f.upload_depth(hardware);
    };
    auto image = [&] {
      const auto bytes = f.read(f.exported.p);
      std::vector<float> values(size_t(width) * 2 * height * 4);
      for (size_t pixel = 0; pixel < size_t(width) * 2 * height; ++pixel) {
        if (f.color == 1) {
          std::uint32_t packed {};
          std::memcpy(&packed, bytes.data() + pixel * 4, 4);
          for (unsigned c = 0; c < 3; ++c) {
            values[pixel * 4 + c] = linear(float((packed >> (10 * c)) & 1023) / 1023.f);
          }
          values[pixel * 4 + 3] = float((packed >> 30) & 3) / 3.f;
        } else {
          for (unsigned c = 0; c < 4; ++c) {
            std::uint16_t half {};
            std::memcpy(&half, bytes.data() + pixel * 8 + c * 2, 2);
            values[pixel * 4 + c] = decode_half(half);
          }
        }
      }
      for (float value : values) {
        need(std::isfinite(value), "Parity export contains nonfinite data");
      }
      return values;
    };
    unsigned settle_limit = 60;
    if (const char *value = std::getenv("SUNSHINE_STEREO_PARITY_SETTLE_FRAMES"); value && *value) {
      size_t used = 0;
      const auto parsed = std::stoul(value, &used);
      need(used == std::strlen(value) && parsed >= 60 && parsed <= 600 && parsed % 3 == 0,
        "Parity settling limit must be a multiple of3 from60 through600");
      settle_limit = unsigned(parsed);
    }
    report << "settling_budget_frames=" << settle_limit << " tolerance_max_linear_rgb=.002 consecutive_batches=2\n";
    if (deterministic) {
      std::ostringstream contract;
      contract << "SUNSHINE_PARITY_TIMELINE_1\nshader_fps=60\nfirst_framecount=1\nfirst_timer_ms=0\n"
        "main_history_before_calibration=0\nfixed_settle_frames=" << settle_limit
        << "\nfinal_capture_included_in_settle_budget=1\nsettle_rgb_tolerance=.002\nfinal_consecutive_batches=2\n"
        "geometry_controls_unchanged=1\nproduction_depth_and_camera_clocks_overridden=0\n";
      const auto value = contract.str();
      write_bytes(output_directory / "timeline-contract.txt", value.data(), value.size());
      if (comparison)
        need(read_file(reference_directory / "timeline-contract.txt") == std::vector<std::uint8_t>(value.begin(), value.end()),
          "Parity deterministic capture schedule differs from the reference");
      report << "shader_clock=fixed_60Hz known_inputs_before_first_stereo=1 static_capture_schedule=fixed_budget\n";
    }
    auto settled = [&](const char *label) {
      const auto before_tick = timeline_tick;
      if (deterministic) {
        // The endpoint is predetermined. Earlier full-frame readbacks cannot
        // select it and only add CPU/GPU stalls; inspect the three complete
        // images defining the unchanged final two-batch stability gate.
        for (unsigned i = 0; i < settle_limit - 6; ++i) f.step();
        auto previous = image();
        unsigned stable = 0;
        for (unsigned batch = 0; batch < 2; ++batch) {
          if (batch == 1) f.measured_frames();
          else for (unsigned i = 0; i < 3; ++i) f.step();
          auto current = image();
          float maximum = 0;
          for (size_t i = 0; i < current.size(); i += 4)
            for (unsigned c = 0; c < 3; ++c)
              maximum = std::max(maximum, std::abs(current[i + c] - previous[i + c]));
          report << "settle case=" << label << " frames=" << settle_limit - 3 + batch * 3
                 << " max_linear_rgb_delta=" << maximum << '\n';
          stable += maximum <= .002f;
          previous = std::move(current);
        }
        record_timeline_capture(label, before_tick);
        need(timeline_tick - before_tick == settle_limit, "Deterministic capture did not follow its fixed frame budget");
        need(stable == 2, "Parity source was not stable across the final scheduled capture batches");
        return previous;
      }
      for (unsigned i = 0; i < 30; ++i) {
        f.step();
      }
      auto previous = image();
      unsigned stable = 0;
      for (unsigned batch = 0; batch < (settle_limit - 30) / 3; ++batch) {
        for (unsigned i = 0; i < 3; ++i) f.step();
        auto current = image();
        float maximum = 0;
        for (size_t i = 0; i < current.size(); i += 4) {
          for (unsigned c = 0; c < 3; ++c) {
            maximum = std::max(maximum, std::abs(current[i + c] - previous[i + c]));
          }
        }
        report << "settle case=" << label << " frames=" << 33 + batch * 3 << " max_linear_rgb_delta=" << maximum << '\n';
        stable = maximum <= .002f ? stable + 1 : 0;
        previous = std::move(current);
        if (stable == 2) {
          f.measured_frames();
          return image();
        }
      }
      throw std::runtime_error("Parity source did not converge within the recorded settling budget");
    };
    auto landmarks = [&](const std::vector<float> &pixels, bool full) {
      shifts result;
      const unsigned top = full ? 0 : marker_top;
      for (unsigned plane = 0; plane < 3; ++plane) {
        for (unsigned eye = 0; eye < 2; ++eye) {
          const unsigned y = top + (2 * plane + 1) * (height - top) / 6;
          const auto at = [&](unsigned x) {
            return pixels[(size_t(y) * width * 2 + eye * width + x) * 4];
          };
          const double background = .5 * (at(width / 4) + at(width * 3 / 4));
          double mass = 0, moment = 0;
          for (unsigned x = width / 4; x < width * 3 / 4; ++x) {
            const double weight = std::max(0., double(at(x)) - background);
            mass += weight;
            moment += weight * (double(x) + .5);
          }
          need(mass > 3 && mass < 20, "Parity landmark missing, clipped or distorted beyond measurement bounds");
          result.eye[plane][eye] = moment / mass - double(width) / 2;
        }
      }
      return result;
    };
    auto save_inputs = [&](const char *name) {
      const std::string color_name = std::string(name) + ".source.bin", depth_name = std::string(name) + ".depth.f32";
      write_bytes(output_directory / color_name, f.source_bytes.data(), f.source_bytes.size());
      write_bytes(output_directory / depth_name, hardware.data(), hardware.size() * sizeof(float));
      report << "input case=" << name << " source_fnv1a64=" << std::hex << fingerprint(f.source_bytes.data(), f.source_bytes.size())
             << " depth_fnv1a64=" << fingerprint(hardware.data(), hardware.size() * sizeof(float)) << std::dec << '\n';
      if (comparison) {
        need(read_file(reference_directory / color_name) == f.source_bytes, "Parity source bytes differ from original reference");
        const auto reference_depth = read_file(reference_directory / depth_name);
        need(reference_depth.size() == hardware.size() * sizeof(float) && std::memcmp(reference_depth.data(), hardware.data(), reference_depth.size()) == 0, "Parity depth bytes differ from original reference");
      }
    };
    build_case(0);
    save_inputs("calibration");
    start_timeline(runtime);
    const auto unit = landmarks(settled("calibration-unit"), true);
    log_shifts(report, "calibration-unit", unit);
    shifts target = unit;
    if (independent) {
      target = read_reference(reference_directory / "calibration.txt", width, height, f.color);
      f.set_float("ScreenPlane", .25f);
      const auto plane_response = landmarks(settled("plane-unit"), true);
      const double slope = (plane_response.disparity(2) - unit.disparity(2)) / .25;
      const double span = unit.disparity(0) - unit.disparity(1);
      need(std::abs(span) > 1 && std::abs(slope) > 1, "Native parity controls have no measurable unit response");
      const double strength = (target.disparity(0) - target.disparity(1)) / span;
      const double plane = (target.disparity(0) / strength - unit.disparity(0)) / slope;
      report << "fitted strength=" << strength << " screen_plane=" << plane << " anchor=.0125 gain=" << 1.f / .015f << '\n';
      need(std::isfinite(strength) && std::isfinite(plane) && strength > 0 && strength <= 2 && plane >= -1 && plane <= 1, "Measured original disparity cannot be matched within native control range/direction");
      f.set_float("Strength", float(strength));
      f.set_float("ScreenPlane", float(plane));
      const auto fitted = landmarks(settled("calibration-fitted"), true);
      log_shifts(report, "calibration-fitted", fitted);
      for (unsigned i = 0; i < 2; ++i) {
        need(std::abs(fitted.disparity(i) - target.disparity(i)) <= .25, "Parity signed endpoint disparity differs by more than 0.25 pixels");
      }
      report << "holdout_binocular_mismatch_px=" << fitted.disparity(2) - target.disparity(2)
             << " classification=depth_transfer_difference_not_endpoint_edge_failure\n";
    } else if (match_automatic && comparison) {
      target = read_reference(reference_directory / "calibration.txt", width, height, f.color);
      const auto fitted = fixed_camera.present ? unit :
        sunshine_matched_disparity::fit(target, [&](double gain, double zero, const char *label) {
          automatic_controls(gain, zero);
          return landmarks(settled(label), true);
        }, report);
      log_shifts(report, "calibration-fitted", fitted);
      if (fixed_camera_reference) {
        const double error = std::max(std::abs(fitted.disparity(0) - target.disparity(0)),
          std::abs(fitted.disparity(1) - target.disparity(1)));
        const double eye_error = sunshine_matched_disparity::endpoint_eye_error(fitted, target);
        report << "fixed_variant_calibration_disparity_error_px=" << error
          << " endpoint_eye_error_px=" << eye_error << " controls_refitted=0\n";
        need(error <= .25 && eye_error <= .25, "Fixed Automatic variant changed calibration endpoint geometry");
      }
      report << "holdout_binocular_mismatch_px=" << fitted.disparity(2) - target.disparity(2)
             << " classification=depth_transfer_difference_not_endpoint_edge_failure\n";
    } else if (comparison) {
      target = read_reference(reference_directory / "calibration.txt", width, height, f.color);
      double largest = 0;
      for (unsigned i = 0; i < 3; ++i) {
        largest = std::max(largest, std::abs(unit.disparity(i) - target.disparity(i)));
      }
      report << "original_variant_calibration_max_disparity_difference_px=" << largest << " controls_refitted=0\n";
      need(largest <= .25, "Original shader variant changed calibration geometry with identical controls");
    } else {
      write_reference(output_directory / "calibration.txt", target, width, height, f.color);
      if (fixed_camera.present)
        report << "calibration_reference=test_only_fixed_automatic_controls controls_refitted=0\n";
    }
    uniforms(runtime, effect, output_directory / "calibrated-uniforms.txt");
    textures(runtime, effect, output_directory / "texture-layouts.txt");

    if (fit_only) {
      // Calibration probes intentionally have their own execution history.
      // Preserve it in this process, then let a fresh fixed-camera treatment
      // run match every original quality-run clock/capture tick exactly.
      float gain = 0, convergence[2] {};
      runtime->get_uniform_value_float(f.uniform("Sunshine_CameraDepthScale"), &gain, 1);
      runtime->get_uniform_value_float(f.uniform("Sunshine_CameraConvergence"), convergence, 2);
      need(std::isfinite(gain) && gain > 0 && gain <= 16000 && std::isfinite(convergence[0]) &&
        convergence[0] > 0 && convergence[0] <= 1 && std::isfinite(convergence[1]) && convergence[1] >= 0 && convergence[1] <= 1,
        "Fitted camera is outside the fixed-camera fixture domain");
      std::ostringstream fitted_input;
      fitted_input << std::setprecision(std::numeric_limits<float>::max_digits10)
        << "# Actual-GPU near/far landmark fit only; no production calibration claim\n"
        << "H=" << gain << "\nreferenceZPD=" << convergence[0] << "\nt0=" << convergence[1] << '\n';
      const auto input_text = fitted_input.str();
      write_bytes(output_directory / "fitted-camera.txt", input_text.data(), input_text.size());
      persist_timeline();
      report << "scope=isolated_landmark_fit_only quality_images_collected=0\n";
      need(report.good(), "Cannot write isolated fit evidence");
      std::puts("PASS isolated actual-GPU camera fit; use fitted-camera.txt in a fresh deterministic treatment process");
      return;
    }

    // Same actual effect submission interval in both runtimes. These borrowed
    // pointers are cleared before the RAII-owned query resources are released.
    decltype(f.exported) timestamp_buffer;
    ID3D12QueryHeap *heap = nullptr;

    struct cleanup_t {
      ID3D12QueryHeap *&heap;

      ~cleanup_t() {
        armed = false;
        queries = nullptr;
        query_readback = nullptr;
        if (heap) {
          heap->Release();
        }
      }
    } cleanup {heap};

    D3D12_QUERY_HEAP_DESC description {};
    description.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
    description.Count = 2;
    need(SUCCEEDED(f.game->CreateQueryHeap(&description, IID_PPV_ARGS(&heap))), "Cannot create parity GPU query heap");
    f.buffer(timestamp_buffer, 2 * sizeof(UINT64), D3D12_HEAP_TYPE_READBACK);
    auto *queue = reinterpret_cast<ID3D12CommandQueue *>(runtime->get_command_queue()->get_native());
    UINT64 frequency {};
    need(SUCCEEDED(queue->GetTimestampFrequency(&frequency)) && frequency, "Cannot obtain parity GPU frequency");
    queries = heap;
    query_readback = timestamp_buffer.p;
    for (unsigned scene = 1; scene <= 3; ++scene) {
      const char *name = scene == 1 ? "step" : scene == 2 ? "detail" :
                                                            "fringe";
      build_case(scene);
      save_inputs(name);
      for (unsigned aa = 0; aa < aa_count; ++aa) {
        set_aa(aa != 0);
        const std::string label = std::string(name) + (aa ? "-aa1" : "-aa0");
        const auto pixels = settled(label.c_str());
        const auto measured = landmarks(pixels, false);
        log_shifts(report, label.c_str(), measured);
        if (comparison) {
          const auto ref = read_reference(reference_directory / (label + ".landmarks.txt"), width, height, f.color);
          double error = 0;
          for (unsigned plane = 0; plane < 2; ++plane) {
            error = std::max(error, std::abs(measured.disparity(plane) - ref.disparity(plane)));
          }
          const double eye_error = sunshine_matched_disparity::endpoint_eye_error(measured, ref);
          all_scene_endpoints_matched = all_scene_endpoints_matched && error <= .25 && eye_error <= .25;
          report << "paired case=" << label << " endpoint_disparity_max_error_px=" << error
                 << " endpoint_eye_max_error_px=" << eye_error
                 << " geometry_matched=" << (error <= .25 && eye_error <= .25) << " holdout_error_px=" << measured.disparity(2) - ref.disparity(2) << '\n';
        } else {
          write_reference(output_directory / (label + ".landmarks.txt"), measured, width, height, f.color);
        }
        // Preserve measured landmarks in both runs for independent artifact
        // validation, including each eye and the held-out middle plane.
        write_reference(output_directory / (label + ".observed-landmarks.txt"), measured, width, height, f.color);
        write_bytes(output_directory / (label + ".rgba.f32"), pixels.data(), pixels.size() * sizeof(float));
        save_preparation(label);
        if (comparison) {
          const auto reference = read_file(reference_directory / (label + ".rgba.f32"));
          need(reference.size() == pixels.size() * sizeof(float), "Parity final pixel layout differs");
          const bool identical = std::memcmp(reference.data(), pixels.data(), reference.size()) == 0;
          double largest_rgb = 0, squared_rgb = 0, largest_alpha = 0;
          for (size_t i = 0; i < pixels.size(); ++i) {
            float expected {};
            std::memcpy(&expected, reference.data() + i * sizeof(float), sizeof(float));
            need(std::isfinite(expected), "Parity reference contains nonfinite pixels");
            const double error = std::abs(double(pixels[i]) - expected);
            if (i % 4 == 3) {
              largest_alpha = std::max(largest_alpha, error);
            } else {
              largest_rgb = std::max(largest_rgb, error);
              squared_rgb += error * error;
            }
          }
          all_scene_pixels_identical = all_scene_pixels_identical && identical;
          report << "pixel_comparison case=" << label << " identical_rgba=" << identical
                 << " max_linear_rgb_error=" << largest_rgb << " rms_linear_rgb_error="
                 << std::sqrt(squared_rgb / (pixels.size() * .75)) << " max_alpha_error=" << largest_alpha
                 << " visible_and_unknown_gap_pixels_combined_descriptive_only=1\n";
        }
        float minimum = INFINITY, maximum = -INFINITY;
        for (size_t i = 0; i < pixels.size(); i += 4) {
          for (unsigned c = 0; c < 3; ++c) {
            minimum = std::min(minimum, pixels[i + c]);
            maximum = std::max(maximum, pixels[i + c]);
          }
        }
        report << "output case=" << label << " linear_rgb_min=" << minimum << " linear_rgb_max=" << maximum
               << " nonfinite=0 input_and_export_no_clamp=1\n";
        uniforms(runtime, effect, output_directory / (label + ".uniforms.txt"));
        // Timing is collected after fixed content has settled and outside all
        // reads/writes. Scene dumps allow separate visible/gap quality analysis.
        for (unsigned i = 0; i < timing_warmup; ++i) {
          f.step();
        }
        std::vector<double> durations(timing_samples);
        for (unsigned sample = 0; sample < durations.size(); ++sample) {
          started = finished = invalid = false;
          armed = true;
          f.step();
          armed = false;
          need(started && finished && !invalid, "Parity GPU bracket did not contain one complete selected technique");
          void *mapped {};
          const D3D12_RANGE range {0, 2 * sizeof(UINT64)};
          need(SUCCEEDED(timestamp_buffer->Map(0, &range, &mapped)), "Cannot read parity GPU timestamps");
          UINT64 ticks[2] {};
          std::memcpy(ticks, mapped, sizeof(ticks));
          const D3D12_RANGE no_write {0, 0};
          timestamp_buffer->Unmap(0, &no_write);
          need(ticks[1] > ticks[0], "Parity GPU timestamps are nonmonotonic");
          durations[sample] = 1000. * double(ticks[1] - ticks[0]) / double(frequency);
          report << "gpu_sample case=" << label << " index=" << sample << " begin=" << ticks[0] << " end=" << ticks[1]
                 << " frequency=" << frequency << " gpu_ms=" << durations[sample] << '\n';
        }
        std::sort(durations.begin(), durations.end());
        const double median = .5 * (durations[(timing_samples - 1) / 2] + durations[timing_samples / 2]);
        const double p95 = durations[(95 * timing_samples + 99) / 100 - 1];
        report << "gpu_result case=" << label << " scope=effects_begin_through_selected_technique_including_COLOR_capture"
               << " warmup=" << timing_warmup << " samples=" << timing_samples << " median_ms=" << median
               << " p95_ms=" << p95 << " p95_method=nearest_rank"
               << " min_ms=" << durations.front() << " max_ms=" << durations.back() << '\n';
      }
    }
    const char *motion_env = std::getenv("SUNSHINE_STEREO_PARITY_MOTION");
    need(!motion_env || !*motion_env || !std::strcmp(motion_env, "0") || !std::strcmp(motion_env, "1"),
      "SUNSHINE_STEREO_PARITY_MOTION must be 0 or 1");
    if (motion_env && !std::strcmp(motion_env, "1")) {
      // Advance actual effects once per moving input. Do not settle each frame:
      // doing so would hide temporal artifacts. The marker strip stays fixed.
      for (unsigned aa = 0; aa < aa_count; ++aa) {
        set_aa(aa != 0);
        build_case(1, -4);
        settled("motion-preroll");
        for (unsigned frame = 0; frame < 7; ++frame) {
          const std::string label = "motion-step-aa" + std::to_string(aa) + "-f" + std::to_string(frame);
          build_case(1, int(frame) - 3);
          save_inputs(label.c_str());
          const auto before_tick = timeline_tick;
          f.step();
          record_timeline_capture(label.c_str(), before_tick);
          const auto pixels = image();
          const auto measured = landmarks(pixels, false);
          log_shifts(report, label.c_str(), measured);
          write_reference(output_directory / (label + ".observed-landmarks.txt"), measured, width, height, f.color);
          write_reference(output_directory / (label + ".landmarks.txt"), measured, width, height, f.color);
          write_bytes(output_directory / (label + ".rgba.f32"), pixels.data(), pixels.size() * sizeof(float));
          save_preparation(label);
          report << "motion case=" << label << " sequence_index=" << frame << " source_offset_px=" << int(frame) - 3
                 << " effect_steps_since_prior_input=1 per_frame_settle=0 wall_clock_cadence_not_controlled=1"
                 << " shader_clock_fixed_60Hz=" << deterministic << '\n';
          if (comparison) {
            const auto ref = read_reference(reference_directory / (label + ".landmarks.txt"), width, height, f.color);
            const double error = sunshine_matched_disparity::endpoint_eye_error(measured, ref);
            const double disparity_error = std::max(std::abs(measured.disparity(0) - ref.disparity(0)),
              std::abs(measured.disparity(1) - ref.disparity(1)));
            all_scene_endpoints_matched &= error <= .25 && disparity_error <= .25;
            report << "paired case=" << label << " endpoint_eye_max_error_px=" << error
                   << " endpoint_disparity_max_error_px=" << disparity_error << '\n';
          }
        }
      }
    }
    report << "comparison_eligibility=" << (comparison ? (all_scene_endpoints_matched ? "matched_endpoints" : "scene_endpoint_mismatch") :
      (fixed_camera.present ? "fixed_automatic_reference" : "original_reference"))
           << " third_plane_transfer_mismatch_reported_separately=1 quality_parity_claim=0\n";
    need(report.good(), "Parity report write failed");
    if (deterministic) {
      persist_timeline();
      if (comparison) {
        need(read_file(reference_directory / "timeline-frames.txt") == read_file(output_directory / "timeline-frames.txt"),
          "Parity shader time/frame input sequence differs from the reference");
        need(read_file(reference_directory / "timeline-captures.txt") == read_file(output_directory / "timeline-captures.txt"),
          "Parity capture ticks differ from the reference");
      }
    }
    need(!comparison || all_scene_endpoints_matched, "Parity evidence collected, but scene endpoint disparity mismatch prevents a matched-quality claim");
    need(!exact_pixels || all_scene_pixels_identical, "Parity evidence collected, but exact renamed-pipeline pixels differ from baseline");
    std::printf("PASS parity evidence captured for %s: %s; no automatic quality-parity claim\n", variant, output_directory.string().c_str());
  }
}  // namespace sunshine_parity

// SPDX-License-Identifier: GPL-3.0-only
// Test-only measured control fitting; this is not a rendering/calibration policy.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace sunshine_matched_disparity {
  struct camera_values {
    bool present = false;
    float gain = 0, reference = 0, zero = 0;
  };

  inline camera_values fixed_camera() {
    const char *path = std::getenv("SUNSHINE_STEREO_PARITY_FIXED_CAMERA");
    if (!path || !*path) return {};
    std::ifstream input(path);
    if (!input.good()) throw std::runtime_error("Cannot read fixed test-only camera input");
    camera_values result;
    result.present = true;
    std::set<std::string> seen;
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line.front() == '#') continue;
      const auto split = line.find('=');
      if (split == std::string::npos) throw std::runtime_error("Fixed camera requires name=value");
      const auto name = line.substr(0, split), text = line.substr(split + 1);
      size_t used = 0;
      const auto value = std::stof(text, &used);
      if (!seen.insert(name).second || used != text.size() || !std::isfinite(value))
        throw std::runtime_error("Invalid fixed test-only camera value");
      if (name == "H") result.gain = value;
      else if (name == "referenceZPD") result.reference = value;
      else if (name == "t0") result.zero = value;
      else throw std::runtime_error("Unknown fixed test-only camera field");
    }
    if (!input.eof() || seen.size() != 3 || result.gain <= 0 || result.gain > 16000 ||
        result.reference <= 0 || result.reference > 1 || result.zero < 0 || result.zero > 1)
      throw std::runtime_error("Fixed test-only camera outside admitted domain");
    return result;
  }

  inline std::array<float, 3> scene_depths() {
    std::array<float, 3> result {.02f, .005f, .0125f}; // Foreground, background, held-out middle.
    const char *path = std::getenv("SUNSHINE_STEREO_PARITY_SCENE");
    if (!path || !*path) return result;
    std::ifstream input(path);
    if (!input.good()) throw std::runtime_error("Cannot open parity scene-depth file");
    std::set<std::string> seen;
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line.front() == '#') continue;
      const auto split = line.find('=');
      if (split == std::string::npos) throw std::runtime_error("Parity scene requires exact name=value lines");
      const auto name = line.substr(0, split), text = line.substr(split + 1);
      size_t used = 0;
      const float value = std::stof(text, &used);
      if (!seen.insert(name).second || used != text.size() || !std::isfinite(value))
        throw std::runtime_error("Invalid, duplicate or nonfinite parity scene depth");
      if (name == "background_raw") result[1] = value;
      else if (name == "middle_raw") result[2] = value;
      else if (name == "foreground_raw" && value == .02f) result[0] = value;
      else throw std::runtime_error("Unknown scene field or changed foreground; the edge oracle requires foreground_raw=0.02");
    }
    if (!input.eof() || seen.empty() || result[1] < 0 || !(result[1] < result[2] && result[2] < result[0]))
      throw std::runtime_error("Parity scene requires 0 <= background_raw < middle_raw < foreground_raw=0.02");
    return result;
  }

  template<class SetFloat, class SetInt, class SetVector, class SetBool, class Log>
  void controls(SetFloat set_float, SetInt set_int, SetVector set_vector, SetBool set_bool, Log &log) {
    const char *path = std::getenv("SUNSHINE_STEREO_PARITY_CONTROLS");
    if (!path || !*path) return;
    std::ifstream input(path);
    if (!input.good()) throw std::runtime_error("Cannot open parity controls file");
    const std::set<std::string> integers {"Range_Boost", "ZPD_Boundary", "WP", "Depth_Map_View",
      "View_Mode", "View_Mode_Warping", "Warping_Masking", "Weapon_Near_Halo_Reduction",
      "Performance_Level", "Reconstruction_Size"};
    const std::set<std::string> floats {"Depth_Adjustment", "Depth_Map_Adjust", "Zero_Parallax_Distance",
      "Auto_Depth_Adjust", "ZPD_Balance", "Compatibility_Power", "ZPD_OverShoot", "Sharpen_Power"};
    std::set<std::string> seen;
    std::string line;
    while (std::getline(input, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      if (line.empty() || line.front() == '#') continue;
      const auto split = line.find('=');
      if (split == std::string::npos) throw std::runtime_error("Parity controls must be exact name=value lines");
      const auto name = line.substr(0, split), text = line.substr(split + 1);
      if (!seen.insert(name).second) throw std::runtime_error("Duplicate parity control");
      if (name == "De_Artifacting") {
        const auto comma = text.find(',');
        if (comma == std::string::npos || text.find(',', comma + 1) != std::string::npos)
          throw std::runtime_error("De_Artifacting requires exactly two comma-separated values");
        std::array<float, 2> values {};
        const std::array<std::string, 2> components {text.substr(0, comma), text.substr(comma + 1)};
        for (size_t index = 0; index < values.size(); ++index) {
          size_t used = 0;
          values[index] = std::stof(components[index], &used);
          if (used != components[index].size() || !std::isfinite(values[index]) || values[index] < -1 || values[index] > 1)
            throw std::runtime_error("De_Artifacting components must be finite values in [-1,1]");
        }
        set_vector(name.c_str(), values.data(), values.size());
        log << "explicit_artistic_control " << name << '=' << values[0] << ',' << values[1] << '\n';
        continue;
      }
      size_t used = 0;
      const double value = std::stod(text, &used);
      if (used != text.size() || !std::isfinite(value))
        throw std::runtime_error("Invalid or nonfinite parity control");
      if (name == "Extended_Smoothing") {
        if (value != 0 && value != 1) throw std::runtime_error("Extended_Smoothing must be 0 or 1");
        set_bool(name.c_str(), value != 0);
      } else if (integers.count(name)) {
        if (value != std::floor(value) || value < -100 || value > 100)
          throw std::runtime_error("Invalid integer parity control");
        const int maximum = name == "View_Mode" ? 5 : name == "View_Mode_Warping" ? 9 :
          name == "Depth_Map_View" || name == "Warping_Masking" || name == "Performance_Level" ? 2 :
          name == "Weapon_Near_Halo_Reduction" || name == "Reconstruction_Size" ? 1 : 100;
        if (maximum != 100 && (value < 0 || value > maximum))
          throw std::runtime_error("Shared parity quality control outside its supported range");
        set_int(name.c_str(), int(value));
      } else if (floats.count(name)) {
        if (name == "Compatibility_Power" ? value < -1 || value > 1 : value < 0 || value > 1000)
          throw std::runtime_error("Invalid float parity control");
        set_float(name.c_str(), float(value));
      } else throw std::runtime_error("Unknown parity control; no arbitrary camera/readiness injection allowed");
      log << "explicit_artistic_control " << name << '=' << value << '\n';
    }
    if (!input.eof()) throw std::runtime_error("Cannot read parity controls file");
  }

  inline bool requested() {
    const char *value = std::getenv("SUNSHINE_STEREO_PARITY_MATCH_AUTOMATIC");
    if (value && *value && std::strcmp(value, "0") && std::strcmp(value, "1"))
      throw std::runtime_error("SUNSHINE_STEREO_PARITY_MATCH_AUTOMATIC must be 0 or 1");
    return value && std::strcmp(value, "1") == 0;
  }

  template<class Shifts>
  double endpoint_eye_error(const Shifts &actual, const Shifts &target) {
    double error = 0;
    for (unsigned plane = 0; plane < 2; ++plane)
      for (unsigned eye = 0; eye < 2; ++eye)
        error = std::max(error, std::abs(actual.eye[plane][eye] - target.eye[plane][eye]));
    return error;
  }

  // Fit only the two interior endpoint landmarks. The third plane and every
  // scene edge are held out. All responses come from actual rendered pixels.
  template<class Shifts, class Render, class Log>
  Shifts fit(const Shifts &target, Render render, Log &log) {
    // H and H*t0 are the independent slope/offset coordinates. Fitting H/t0
    // directly produces an avoidable bilinear Newton overshoot when the
    // reference is much stronger than the initial probe.
    double gain = 128, offset = 128 * .0125;
    for (unsigned iteration = 0; iteration < 8; ++iteration) {
      const double zero = offset / gain;
      const auto actual = render(gain, zero, "automatic-fit-base");
      const double e0 = target.disparity(0) - actual.disparity(0);
      const double e1 = target.disparity(1) - actual.disparity(1);
      log << "automatic_fit iteration=" << iteration << " H=" << gain << " t0=" << zero
          << " near_error_px=" << e0 << " far_error_px=" << e1
          << " max_endpoint_eye_error_px=" << endpoint_eye_error(actual, target) << '\n';
      if (std::max(std::abs(e0), std::abs(e1)) <= .125 && endpoint_eye_error(actual, target) <= .125)
        return actual;
      const double dg = std::max(8., gain * .25), dz = 1;
      const auto gain_probe = render(gain + dg, offset / (gain + dg), "automatic-fit-gain-probe");
      const auto zero_probe = render(gain, (offset + dz) / gain, "automatic-fit-zero-probe");
      const double a = (gain_probe.disparity(0) - actual.disparity(0)) / dg;
      const double b = (zero_probe.disparity(0) - actual.disparity(0)) / dz;
      const double c = (gain_probe.disparity(1) - actual.disparity(1)) / dg;
      const double d = (zero_probe.disparity(1) - actual.disparity(1)) / dz;
      const double determinant = a * d - b * c;
      log << "automatic_fit_jacobian dnear_dH=" << a << " dnear_du=" << b
          << " dfar_dH=" << c << " dfar_du=" << d << " determinant=" << determinant << '\n';
      if (!std::isfinite(determinant) || std::abs(determinant) < 1e-7)
        throw std::runtime_error("Automatic measured landmark fit is singular/clamped; no matched comparison");
      gain += (d * e0 - b * e1) / determinant;
      offset += (a * e1 - c * e0) / determinant;
      log << "automatic_fit_proposal H=" << gain << " u=" << offset << " t0=" << offset / gain << '\n';
      if (!std::isfinite(gain) || !std::isfinite(offset) || gain <= 0 || gain > 16000 || offset < 0 || offset / gain > 1)
        throw std::runtime_error("Manual endpoint disparity lies outside the admitted Automatic fit domain");
    }
    throw std::runtime_error("Automatic pixel landmark fit did not converge; no matched comparison");
  }
}

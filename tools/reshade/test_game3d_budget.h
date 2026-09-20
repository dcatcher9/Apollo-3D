// SPDX-License-Identifier: GPL-3.0-only
#pragma once

// Shared assertions over actual native GPU readbacks, not a CPU warp replica.
#include "game3d_renderer.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sunshine_game3d_test {
  struct budget_case {
    const char *name;
    float strength, blend, limit;
  };

  inline std::vector<budget_case> budget_cases() {
    const float limit = sunshine_game3d::default_disparity_limit_uv;
    return {{"strength0", 0, 1, limit}, {"strength1", 1, 1, limit},
      {"strength50", 50, 1, limit}, {"strength100", 100, 1, limit},
      {"reentry35", 50, .35f, limit}, {"missing-budget", 100, 1, 0},
      {"invalid-budget", 100, 1, std::numeric_limits<float>::quiet_NaN()},
      {"recovered-budget", 1, 1, limit}};
  }

  inline sunshine_game3d::render_parameters budget_parameters(const budget_case &test) {
    sunshine_game3d::render_parameters p;
    p.depth_ready = p.camera_ready = 1;
    p.coordinate_basis = 1;
    p.projection = {0, 1};
    // A stale gain amplifies both sides of this zero plane far beyond the
    // display budget on the fixtures' existing near/far depth pattern.
    p.depth_scale = 32000;
    p.convergence = {.05f, .02f};
    p.strength = test.strength;
    p.strength_blend = test.blend;
    p.disparity_limit_uv = test.limit;
    return p;
  }

  inline void budget_check(bool value, const std::string &message) {
    if (!value) throw std::runtime_error(message);
  }

  struct field_extrema {
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
  };

  inline field_extrema check_budget_field(const std::vector<std::uint8_t> &bytes,
      unsigned width, unsigned height, double limit, const std::string &label) {
    budget_check(bytes.size() == size_t(width) * height * sizeof(float), label + " is not a full R32 field");
    const double tolerance = std::max(2e-9, limit * 8 * std::numeric_limits<float>::epsilon());
    field_extrema result;
    for (size_t index = 0; index < size_t(width) * height; ++index) {
      float value;
      std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(value));
      if (!std::isfinite(value) || std::abs(double(value)) > limit + tolerance || (limit == 0 && value != 0)) {
        throw std::runtime_error(label + " violates displacement budget at (" +
          std::to_string(index % width) + "," + std::to_string(index / width) +
          "): value=" + std::to_string(value) + " limit=" + std::to_string(limit));
      }
      result.minimum = std::min(result.minimum, value);
      result.maximum = std::max(result.maximum, value);
    }
    if (limit > 0) {
      budget_check(result.minimum <= -limit + tolerance && result.maximum >= limit - tolerance,
        label + " did not exercise both signed displacement caps");
    }
    return result;
  }

  inline void verify_budget_fields(const budget_case &test, unsigned width, unsigned height,
      const std::vector<std::uint8_t> &candidate, const std::vector<std::uint8_t> &final_field,
      std::ostream &report) {
    const double limit = std::isfinite(test.limit) && test.limit > 0 ?
      double(test.limit) * (double(test.strength) / 100) * double(test.blend) : 0;
    const auto before = check_budget_field(candidate, width, height, limit, std::string(test.name) + " candidate");
    const auto after = check_budget_field(final_field, width, height, limit, std::string(test.name) + " final");
    report << "budget " << test.name << " limit_uv=" << limit << " candidate_min=" << before.minimum <<
      " candidate_max=" << before.maximum << " final_min=" << after.minimum << " final_max=" << after.maximum << '\n';
    std::printf("MEASURE budget %s limit_uv=%.9g candidate=[%.9g,%.9g] final=[%.9g,%.9g]\n",
      test.name, limit, before.minimum, before.maximum, after.minimum, after.maximum);
  }

  inline void verify_budget_dump(const std::filesystem::path &directory, float expected_budget) {
    std::ifstream input(directory / "manifest.json");
    budget_check(input.good(), "Displacement budget dump manifest is missing");
    const auto manifest = nlohmann::json::parse(input);
    const auto &metadata = manifest.at("producer_metadata");
    const auto &replay = metadata.at("replay");
    budget_check(replay.at("parameter_abi") == "sunshine_game3d.render_parameters.v2" &&
      replay.at("parameter_bytes") == 80, "Displacement budget dump has the wrong constant ABI");
    bool found = false;
    for (const auto &field : replay.at("parameter_layout")) {
      if (field.at("name") != "disparity_limit_uv") continue;
      budget_check(!found && field.at("byte_offset") == 28 && field.at("count") == 1 &&
        field.at("scalar_type") == "float32", "Displacement budget dump has the wrong byte28 layout");
      found = true;
    }
    budget_check(found && metadata.at("render_parameters").at("disparity_limit_uv").get<float>() == expected_budget,
      "Displacement budget is absent or changed in dump parameters");
    const auto hex = replay.at("parameter_hex").get<std::string>();
    budget_check(hex.size() == 160, "Displacement budget dump lost exact parameter bytes");
    std::uint8_t bytes[sizeof(float)];
    for (size_t i = 0; i < sizeof(float); ++i)
      bytes[i] = static_cast<std::uint8_t>(std::stoul(hex.substr((28 + i) * 2, 2), nullptr, 16));
    budget_check(std::memcmp(bytes, &expected_budget, sizeof(expected_budget)) == 0,
      "Displacement budget dump byte28 differs from consumed GPU constants");
  }
}

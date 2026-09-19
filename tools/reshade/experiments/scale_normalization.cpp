// SPDX-License-Identifier: GPL-3.0-only
// Offline scalar-policy experiment, not a renderer or a shipping gain policy.
// Reuses the real statistics, content classifier and temporal gain controller.
#include "../raw_reference_statistics.h"
#include "../depth_selection_policy.h"
#include "../scene_gain.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: scale_normalization cases.tsv grids.f32 output.csv");
    std::ifstream cases(argv[1]), grids(argv[2], std::ios::binary);
    std::ofstream output(argv[3]);
    if (!cases || !grids || !output) throw std::runtime_error("Cannot open experiment files");
    output << std::setprecision(17)
      << "id,sequence,time_ms,group,phase,method,gate,content_kind,admitted,mean,center,sigma,mad,span,effective_span,contributors,target,current,ready,accepted_target,core_separation,scene_span,output_sigma\n";
    const char *methods[] {"center", "variance", "bounded_1.5", "bounded_2", "bounded_3", "effective_span", "scene_mean"};
    std::map<std::string, sunshine_scene_gain::policy> controllers;
    std::string line;
    std::getline(cases, line);
    unsigned count = 0;
    while (std::getline(cases, line)) {
      std::istringstream row(line);
      unsigned id;
      std::string sequence, group, phase;
      std::uint64_t tick;
      double core_far, core_near;
      if (!(row >> id >> sequence >> tick >> group >> phase >> core_far >> core_near))
        throw std::runtime_error("Malformed case row");
      sunshine_raw_reference::grid grid;
      grids.seekg(std::streamoff(id) * sizeof(grid));
      grids.read(reinterpret_cast<char *>(grid.data()), sizeof(grid));
      if (!grids) throw std::runtime_error("Truncated grid input");
      const auto stats = sunshine_raw_reference::measure(grid);
      const auto content = sunshine_depth::analyze_depth(grid.data(), grid.size(), 32, 18);
      double mad = 0, low = 1, high = 0;
      bool center_interior = true;
      for (unsigned i = 0; i < grid.size(); ++i) {
        mad += std::abs(double(grid[i]) - stats.mean) / grid.size();
        low = std::min(low, double(grid[i])); high = std::max(high, double(grid[i]));
        if (i % 32 >= 14 && i % 32 < 18 && i / 32 >= 7 && i / 32 < 11)
          center_interior &= std::isfinite(grid[i]) && grid[i] > 0 && grid[i] < 1;
      }
      const double span = high - low;
      const double effective = mad > 0 ? 2 * stats.variance / mad : 0;
      const double sigma_gain = stats.sigma > 0 ? .5 / stats.sigma : 0;
      const double targets[] {
        stats.center_mean > 0 ? 1 / stats.center_mean : 0,
        sigma_gain,
        span > 0 ? std::min(sigma_gain, 1.5 / span) : 0,
        span > 0 ? std::min(sigma_gain, 2 / span) : 0,
        span > 0 ? std::min(sigma_gain, 3 / span) : 0,
        effective > 0 ? 1 / effective : 0,
        stats.mean > 0 ? 1 / stats.mean : 0
      };
      for (unsigned method = 0; method < 7; ++method) for (unsigned gate = 0; gate < 2; ++gate) {
        const char *gate_name = gate ? "existing_content" : "finite_only";
        auto &controller = controllers[sequence + '/' + methods[method] + '/' + gate_name];
        const double target = targets[method];
        // Both gate alternatives are visible. Neither labels source confidence
        // as proof that a scene is representative for artistic normalization.
        const bool admitted = stats.valid && center_interior && target > 0 && std::isfinite(target) &&
          (!gate || content.kind == sunshine_depth::content_kind::useful);
        controller.advance(tick);
        if (admitted) {
          controller.observe(1 / target, tick, [](float gain, double) {
            // Same full raw [0,1] preparation storage bound as production.
            return 1.0 + gain <= 16384.0;
          });
        } else controller.invalidate();
        const double current = controller.value();
        output << id << ',' << sequence << ',' << tick << ',' << group << ',' << phase << ','
          << methods[method] << ',' << gate_name << ',' << unsigned(content.kind) << ',' << admitted << ','
          << stats.mean << ',' << stats.center_mean << ',' << stats.sigma << ',' << mad << ',' << span << ','
          << effective << ',' << stats.effective_contributors << ',' << target << ',' << current << ','
          << controller.initialized() << ',' << controller.target() << ',' << current * (core_near-core_far) << ','
          << current * span << ',' << current * stats.sigma << '\n';
      }
      ++count;
    }
    if (!output.good()) throw std::runtime_error("Cannot write results");
    std::cout << "PASS: " << count << " input grids; 7 formulas x 2 explicit admission policies; actual scene_gain.h controller\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
  }
}

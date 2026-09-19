// SPDX-License-Identifier: GPL-3.0-only
// Independent analytic silhouette oracle through the actual ReShade runtime.
// Flat opaque surfaces have known translated bounds; no CPU warp is reproduced.
#define SUNSHINE_NATIVE_STEREO_RUNTIME_HELPER
#include "test_native_stereo_runtime.cpp"

namespace {
  void check_halo(fixture_t &fixture, const fs::path &output) {
    fixture.discover();
    std::vector<float> raw(size_t(width) * height);
    std::ofstream report(output / "halo-measurements.csv");
    report << "shape,aa,shift,max_error,mean_edge_error,wrong_pixels,samples\n";
    bool passed = true;
    for (unsigned shape = 0; shape < 3; ++shape) {
      const auto bounds = [shape](unsigned y) {
        const int edge = int(width / 2) + (int(y) - int(height / 2)) / 4;
        return shape == 0 ? std::array<int, 2>{0, edge} :
          shape == 1 ? std::array<int, 2>{edge, int(width)} :
          std::array<int, 2>{edge, edge + 1};
      };
      for (unsigned y = 0; y < height; ++y) {
        const auto area = bounds(y);
        for (unsigned x = 0; x < width; ++x) {
          const size_t index = size_t(y) * width + x;
          const bool foreground = int(x) >= area[0] && int(x) < area[1];
          raw[index] = foreground ? .8f : .2f;
          // Exact linear scRGB levels: one opaque white foreground and a dark
          // constant background. Any mixed noncoverage fringe is measurable.
          const std::uint16_t value = foreground ? 0x3c00 : 0x2c00; // 1 or 1/16
          const std::uint16_t color[]{value, value, value, 0x3c00};
          std::memcpy(fixture.source_bytes.data() + index * 8, color, sizeof(color));
        }
      }
      fixture.fill_upload(fixture.source_upload, fixture.backbuffers[0]->GetDesc(),
        fixture.source_bytes.data(), fixture.source_footprint);
      fixture.upload_depth(raw);
      fixture.set_float("Sunshine_RawAnchor", .5f);
      fixture.set_float("Sunshine_RawGain", 2.f);
      fixture.set_float("ScreenPlane", 0);
      fixture.set_bool("DepthView", false);
      for (unsigned aa = 0; aa < 2; ++aa) {
        fixture.set_bool("EdgeAntialias", aa != 0);
        for (float phase : {.2f, .5f, .8f}) {
          const float requested_shift = std::floor(.003f * width) + phase;
          const float strength = requested_shift / (.003f * width);
          require(strength > 0 && strength <= 2, "Halo phase exceeds production strength limits");
          fixture.set_float("Strength", strength);
          fixture.measured_frames();
          const auto pixels = fixture.read(fixture.exported.p);
          // Use the same declared camera, not an inferred displacement fitted
          // from the output being scored. This is the constant-plane solution.
          const float shift = (.8f - .5f) * 2.f * (.005f * width * strength);
          double error_sum = 0;
          float maximum = 0;
          size_t samples = 0, wrong = 0;
          for (unsigned y = height / 4; y < height * 3 / 4; y += 3) {
            const auto area = bounds(y);
            const int edge = shape == 0 ? area[1] : area[0];
            for (unsigned eye = 0; eye < 2; ++eye) {
              const float eye_sign = eye == 0 ? 1.f : -1.f;
              const float left = float(area[0]) + eye_sign * shift;
              const float right = float(area[1]) + eye_sign * shift;
              const auto visible_foreground = [left, right](float destination) {
                return destination >= left && destination < right ? 1.f : 0.f;
              };
              for (int x = edge - int(std::ceil(shift)) - 4; x <= edge + int(std::ceil(shift)) + 5; ++x) {
                const float center = float(x) + .5f;
                const float coverage = aa ? .5f * (visible_foreground(center - .25f) + visible_foreground(center + .25f)) :
                  visible_foreground(center);
                const float expected = .0625f + .9375f * coverage;
                const float actual = fixture.channel(pixels, unsigned(x) + eye * width, y, 0);
                require(std::isfinite(actual), "Nonfinite halo output");
                // With AA off, a ray exactly on a zero-area silhouette boundary
                // may resolve either adjacent opaque surface. It must still not
                // blend their colors. Subpixel AA rays here do not land on ties.
                const bool tie = !aa && (std::abs(center - left) < .0001f || std::abs(center - right) < .0001f);
                const float error = tie ? std::min(std::abs(actual - .0625f), std::abs(actual - 1.f)) : std::abs(actual - expected);
                maximum = std::max(maximum, error);
                error_sum += error;
                wrong += error > .003f;
                ++samples;
              }
            }
          }
          report << shape << ',' << aa << ',' << shift << ',' << maximum << ',' <<
            error_sum / double(samples) << ',' << wrong << ',' << samples << '\n';
          std::printf("MEASURE halo shape=%u AA=%u shift=%.6f max=%.9g mean=%.9g wrong=%zu/%zu\n",
            shape, aa, shift, maximum, error_sum / double(samples), wrong, samples);
          passed = passed && wrong == 0;
        }
      }
    }
    require(report.good(), "Could not write halo measurements");
    require(passed, "Cross-surface color halo or silhouette coverage mismatch");
    std::puts("PASS actual scRGB opaque silhouettes: mirrored slopes and one-column foreground, three phases, AA off/on, exact mono preservation");
  }
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc != 4) {
    std::fputs("usage: reshade_native_halo_runtime_test <official-ReShade64.dll> <Sunshine-Shaders> <fresh-output-directory>\n", stderr);
    return 2;
  }
  std::thread([] {
    Sleep(120000);
    std::fputs("FAIL halo fixture watchdog\n", stderr);
    TerminateProcess(GetCurrentProcess(), 124);
  }).detach();
  try {
    width = 1280;
    height = 720;
    fixture_t fixture;
    const auto output = fs::absolute(argv[3]);
    fixture.initialize(fs::absolute(argv[1]), fs::absolute(argv[2]), output, 2, 0);
    check_halo(fixture, output);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL halo fixture: %s\n", error.what());
    return 1;
  }
}

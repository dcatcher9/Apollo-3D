// SPDX-License-Identifier: GPL-3.0-only
#include "projection_depth_scale.h"

#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_projection_depth;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  void close(double actual, double expected, double tolerance, const char *message) {
    require(std::isfinite(actual) && std::abs(actual-expected) <= tolerance, message);
  }
  struct encoding { double A{}, B{}; };
  encoding perspective(bool reversed, double near_plane, double far_plane = 0.0) {
    if (!far_plane) return reversed ? encoding{0.0, near_plane} : encoding{1.0, -near_plane};
    const double denominator = far_plane-near_plane;
    return reversed ? encoding{-near_plane/denominator, near_plane*far_plane/denominator} :
      encoding{far_plane/denominator, -near_plane*far_plane/denominator};
  }
  coefficients build(encoding value) { return make(value.A, value.B); }
  float raw(encoding value, double z) { return static_cast<float>(value.A+value.B/z); }
  float converted(const coefficients &value, float depth) {
    float q = -1;
    require(inverse_distance(value, depth, q), "Known perspective depth failed physical conversion");
    return q;
  }
  float prepared(float K, float q) { return 1.f/(1.f+K*q); }
  float field(float K, float q0, float q) { return reference_zpd*K*(q0-q); }

  void conventions_and_full_depth_domain() {
    for (const bool reversed : {false, true}) for (const double far_plane : {0.0, 1000.0}) {
      const auto encoded = perspective(reversed, .06, far_plane);
      const auto value = build(encoded);
      require(value.valid(), "Normal/reversed finite/infinite perspective was rejected");
      close(value.near_plane, .06, 1e-12, "Near distance did not come from projection endpoints");
      constexpr float gain = 8.f; // Explicit scene reference q=.125; no gain is encoded in the matrix.
      require(supports_scale(value, gain), "Representable scene gain was rejected");
      for (const double z : {.06, .12, 1.0, 10.0, 100.0}) {
        const float q = converted(value, raw(encoded, z));
        close(q, 1.0/z, 2e-6, "Raw depth did not convert to inverse game distance");
        close(prepared(gain,q), 1.0/(1.0+gain/z), 3e-5, "Exact camera reconstruction changed prepared depth");
      }
      for (unsigned i=0; i<=4096; ++i) {
        const float q = converted(value, float(i)/4096.f);
        require(std::isfinite(field(gain,.125f,q)),
          "Some hardware-depth input exceeded the checked FP32 displacement domain");
      }
      close(converted(value, reversed ? 0.f : 1.f), far_plane ? 1.0/far_plane : 0.0, 2e-6,
        "Finite far/infinity was discarded or forced to another depth");
    }
  }

  void independent_gain_zero_and_clipping() {
    const auto encoded = perspective(true, .06, 20000.0);
    const auto value = build(encoded);
    constexpr float K=8.f;
    const float near_q = converted(value, raw(encoded,1.0)), far_q = converted(value,raw(encoded,20.0));
    const float span = field(K,.5f,near_q)-field(K,.5f,far_q);
    for (const double center_z : {.1,1.0,10.0,1000.0}) {
      const float q0 = converted(value,raw(encoded,center_z));
      require(supports_zero(value,K,q0), "Representable screen plane rejected");
      close(field(K,q0,near_q)-field(K,q0,far_q),span,1e-6,
        "Moving the screen plane changed pairwise stereo strength");
      close(field(K,q0,q0),0,0,"The screen plane acquired nonzero disparity");
    }
    for (double near_plane : {.03,.12,1.0}) {
      const auto later_encoding=perspective(true,near_plane,20000.0);
      const auto later=build(later_encoding);
      const auto q=converted(later,raw(later_encoding,8.0));
      close(q,.125,2e-7,"Changing clipping changed reconstructed distance");
      close(prepared(K,q),.5,2e-7,"Clipping implicitly changed artistic gain");
      close(field(K,.1f,q),field(K,.1f,.125f),2e-7,"Clipping implicitly changed stereo strength");
    }
  }

  void unit_and_storage_invariance() {
    const auto encoded=perspective(true,.0625,1024.0);
    const auto original=build(encoded);
    constexpr float K=8.f;
    for (double unit : {.001,100.0,1e6}) {
      const auto transformed=perspective(true,.0625*unit,1024.0*unit);
      const auto value=build(transformed);
      const auto center=converted(value,raw(transformed,8.0*unit));
      const auto adapted_K=1.f/center;
      for (double z : {.125,1.0,16.0,512.0}) {
        const auto q=converted(value,raw(transformed,z*unit));
        const auto original_q=converted(original,raw(encoded,z));
        close(prepared(adapted_K,q),prepared(K,original_q),2e-7,"World-unit change altered prepared depth");
        close(field(adapted_K,float(.125/unit),q),field(K,.125f,original_q),2e-5,
          "World-unit change altered stereo displacement");
      }
    }
    struct transform { double scale,bias; };
    for (bool reversed : {false,true}) for (double far_plane : {0.0,1024.0}) {
      const auto encoded=perspective(reversed,.0625,far_plane);
      const auto original=build(encoded);
      for (const auto storage : std::array<transform,5>{{{.5,.25},{2,-.5},{-1,1},{1,.25},{4,-1}}}) {
        const auto value=make(encoded.A,encoded.B,storage.scale,storage.bias);
        require(value.valid() && value.near_plane==original.near_plane,"Storage changed reconstructed near");
        for (double z : {.0625,.125,1.0,16.0,512.0}) {
          const float stored=float((encoded.A+encoded.B/z-storage.bias)/storage.scale);
          const auto q=converted(value,stored), expected=converted(original,raw(encoded,z));
          close(q,expected,3e-6,"Resource precision changed inverse distance");
          close(prepared(K,q),prepared(K,expected),2e-5,"Resource precision changed prepared depth");
          close(field(K,.25f,q),field(K,.25f,expected),3e-6,"Resource precision changed stereo displacement");
        }
        const float largest_q=std::max(converted(value,value.raw_min),converted(value,value.raw_max));
        require(supports_scale(value,32768.f/largest_q) &&
          supports_scale(value,std::numeric_limits<float>::max()*.125f),
          "Finite displacement retained a reciprocal or FP16 limit");
        float q=123;
        require(!inverse_distance(value,std::nextafter(value.raw_min,-INFINITY),q) && q==0 &&
          !inverse_distance(value,std::nextafter(value.raw_max,INFINITY),q),"Outside-domain pixels admitted");
      }
    }
  }

  void invalid_and_unrepresentable_inputs() {
    for (const auto malformed : std::array<encoding,6>{{{0,0},{.5,1},{.5,-1},{INFINITY,1},{0,NAN},{NAN,.06}}})
      require(!build(malformed).valid(),"Malformed or negative-distance projection admitted");
    for (double near_plane : {1e-100,1e100}) {
      const auto value=build(perspective(true,near_plane));
      require(value.reason==status::unsupported_shader_domain && value.inverseB==0,"Unsupported units silently clamped");
    }
    for (const auto flush_sensitive : std::array<encoding,2>{{{-1e38,1e38},{-1e-40,1}}})
      require(!build(flush_sensitive).valid(),"Flush-sensitive projection coefficient admitted");
    const auto value=make(0,.0625);
    for (float input : {-.001f,1.001f,INFINITY,NAN}) {
      float q=123;
      require(!inverse_distance(value,input,q) && q==0,"Malformed depth clamped or previous coordinate retained");
    }
    require(supports_zero(value,8,0) && supports_zero(value,8,100) && supports_zero(value,8,1e8),
      "Representable zero outside clipping retained a FP16 limit");
    for (double q0 : std::array<double,5>{{-1.0,INFINITY,NAN,double(std::numeric_limits<float>::denorm_min()),1e100}})
      require(!supports_zero(value,8,q0),"Invalid/underflowing/overflowing screen plane admitted");
    require(!supports_scale(value,0) && !supports_scale(value,INFINITY) && !supports_zero(value,0,.1),
      "Unavailable artistic gain authorized rendering");
    for (const auto storage : std::array<encoding,5>{{{0,0},{NAN,0},{1,NAN},{INFINITY,0},{1,INFINITY}}})
      require(!make(0,.0625,storage.A,storage.B).valid(),"Invalid resource precision admitted");
    require(!make(0,.0625,1e100,0).valid() && !make(0,.0625,1,1e30).valid(),
      "Unrepresentable transformed raw coordinates admitted");
  }

  void fp32_displacement_limits() {
    const auto ordinary=make(0,1), close_clip=make(0,.001);
    const float maximum=std::numeric_limits<float>::max();
    require(ordinary.valid() && close_clip.valid(),"Overflow fixture has invalid projection coefficients");
    require(supports_scale(ordinary,maximum) && supports_zero(ordinary,maximum,1),
      "Finite .05*K displacement was rejected because K*q has a larger range");
    require(!supports_scale(close_clip,maximum) && !supports_zero(close_clip,maximum,0),
      "True FP32 near-depth displacement overflow was admitted");
    require(!supports_zero(ordinary,maximum,1000),"True FP32 zero-plane displacement overflow was admitted");
    require(!supports_scale(ordinary,std::numeric_limits<float>::denorm_min()),
      "Flush-sensitive scene gain admitted");
  }
}
int main() {
  try {
    conventions_and_full_depth_domain();
    independent_gain_zero_and_clipping();
    unit_and_storage_invariance();
    invalid_and_unrepresentable_inputs();
    fp32_displacement_limits();
    std::puts("PASS projection geometry: exact depth, independent artistic gain, clipping/units/storage and shader limits");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr,"FAIL projection geometry: %s\n",error.what());
    return 1;
  }
}

// SPDX-License-Identifier: GPL-3.0-only
#include "scene_gain.h"
#include "projection_depth_controller.h"
#include "raw_scene_policy.h"

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
  using sunshine_scene_gain::depth_range;
  using sunshine_depth_statistics::moments;
  void require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
  }
  void close(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > tolerance) {
      std::fprintf(stderr, "%s: actual=%.17g expected=%.17g\n", message, actual, expected);
      throw std::runtime_error(message);
    }
  }
  struct measurement {
    float raw_min{}, raw_max{}, A{}, inverseB{1.f};
    moments statistics;
    explicit measurement(std::vector<float> raw, float offset=0.f, float scale=1.f) :
        A(offset), inverseB(scale) {
      require(!raw.empty(), "Empty measurement fixture");
      raw_min = *std::min_element(raw.begin(), raw.end());
      raw_max = *std::max_element(raw.begin(), raw.end());
      auto &m=statistics;
      m.supplied=m.valid=m.centered_supplied=true;
      m.A=A; m.inverseB=inverseB; m.count=raw.size(); m.tiles_x=1; m.tiles_y=1;
      std::vector<float> decoded;
      for (const float value: raw) decoded.push_back((value-A)*inverseB);
      m.center=*std::min_element(decoded.begin(), decoded.end());
      for (const double q: decoded) {
        m.sum+=q; m.sum_squares+=q*q;
        const double contrast=q-m.center;
        m.sum_centered+=contrast;
        m.sum_centered_squares+=contrast*contrast;
      }
    }
    depth_range decode() const {
      std::array<float, 8> selector;
      selector.fill(std::numeric_limits<float>::quiet_NaN());
      return sunshine_scene_gain::decode_range(selector, true, true, raw_min, raw_max,
        A, inverseB, 0.f, 1.f, statistics);
    }
  };
  struct fixture {
    sunshine_scene_gain::policy value;
    fixture() { value.configure({.5, 1., 1.}); }
    void capture(const depth_range &range, std::uint64_t now) {
      value.advance(now); value.observe(range, now); value.update(now);
    }
    void start(const depth_range &range) {
      for (std::uint64_t now=1000; now<=1750; now+=250) {
        capture(range, now);
        require(value.initialized()==(now==1750), "Centered distribution bypassed startup spacing");
      }
    }
  };

  void distribution_changes_zero_without_changing_gain() {
    // Hand calculation: sum(d)=1.5, sum(d^2)=1.125 -> zero=.375.
    const auto populated=measurement({0.f, .25f, .25f, 1.f}).decode();
    const auto endpoints=measurement({0.f, 0.f, 0.f, 1.f}).decode();
    require(populated.valid() && populated.has_centered_moments, "Full centered distribution was not admitted");
    close(populated.zero_target(), .375, 0., "Zero used extrema or raw mean instead of foreground contrast");
    close(endpoints.zero_target(), .5, 0., "Sparse two-layer foreground was converged onto the screen");
    fixture a,b; a.start(populated); b.start(endpoints);
    close(a.value.zero(), .375, 0., "Startup did not seed contrast zero");
    close(b.value.zero(), .5, 0., "Two-layer startup lost balanced placement");
    close(a.value.ui_midpoint(), .5, 0., "UI did not retain the independent legacy midpoint");
    close(a.value.target_ui_midpoint(), .5, 0., "UI target used contrast-weighted foreground depth");
    require(b.value.ui_midpoint()==b.value.zero(), "Two-layer UI and scene midpoint unexpectedly diverged");
    require(a.value.value()==b.value.value() && a.value.value()==1.f,
      "Changing zero distribution changed independent maximum-reference gain");
    for (const unsigned count: {1u, 5u, 99u}) {
      std::vector<float> raw(100, .125f);
      std::fill(raw.begin(), raw.begin()+count, .75f);
      const auto r=measurement(raw).decode();
      close(r.zero_target(), .4375, 0., "Two-layer occupancy changed its target zero");
    }
    // Missing distribution remains a labeled compatibility path, not invented moments.
    auto legacy=measurement({0.f, .25f, .25f, 1.f});
    legacy.statistics.centered_supplied=false;
    require(!legacy.decode().has_centered_moments, "Legacy payload fabricated centered statistics");
    close(legacy.decode().zero_target(), .5, 0., "Legacy range-only compatibility target changed");
  }

  void centered_payload_corruption_never_falls_back() {
    const measurement original({.125f, .25f, .5f, .75f});
    require(original.decode().valid(), "Valid shifted-center fixture rejected");
    for (unsigned failure=0; failure!=14; ++failure) {
      auto bad=original;
      auto &m=bad.statistics;
      switch (failure) {
        case 0: m.center=0.; break;
        case 1: m.center=std::numeric_limits<double>::quiet_NaN(); break;
        case 2: m.sum_centered=-1.; break;
        case 3: m.sum_centered_squares=-1.; break;
        case 4: m.sum_centered=std::numeric_limits<double>::infinity(); break;
        case 5: m.sum_centered_squares=std::numeric_limits<double>::quiet_NaN(); break;
        case 6: m.sum_centered=0.; break;
        case 7: m.sum_centered_squares=0.; break;
        case 8: m.sum_centered_squares*=100.; break;
        case 9: m.sum_centered_squares*=.01; break;
        case 10: m.sum_centered*=.5; m.sum_centered_squares*=.25; break;
        case 11: m.supplied=false; break;
        case 12: m.inverseB=-1.f; break;
        // Still inside centered-moment bounds, but contradicts the separately
        // supplied raw second moment and must not authorize a different zero.
        case 13: m.sum_centered_squares*=.8; break;
      }
      require(!bad.decode().valid(), "Malformed centered payload silently fell back or was accepted");
    }
  }

  void near_flat_uses_centered_evidence_across_units() {
    const float low=1.f-1e-6f;
    for (const float scale: {1e-30f, 1.f, 1e30f}) {
      auto source=measurement({low, 1.f}, 0.f, scale);
      const auto range=source.decode();
      require(range.valid() && range.has_centered_moments, "Representable near-flat centered evidence was rejected");
      close(range.zero_target()/scale, (range.minimum/scale+range.maximum/scale)*.5, 1e-14,
        "Near-flat target lost centered detail under unit changes");
      fixture initialized; initialized.start(range);
      require(std::isnormal(initialized.value.value()) && initialized.value.zero()>=range.minimum &&
        initialized.value.zero()<=range.maximum, "Extreme units produced unsupported gain or out-of-range zero");
    }
    // Reproduce lossy raw diagnostic moments: these remain acceptable rounded
    // diagnostics, while trustworthy centered moments determine the zero.
    auto lossy=measurement({low, 1.f});
    const float rounded_mean=(low+1.f)*.5f;
    const float rounded_square=(low*low+1.f)*.5f;
    lossy.statistics.sum=double(rounded_mean)*2.;
    lossy.statistics.sum_squares=double(rounded_square)*2.;
    const auto decoded=lossy.decode();
    require(decoded.valid(), "Rounded raw diagnostics vetoed stable near-flat evidence");
    close(decoded.zero_target(), (double(low)+1.)*.5, 0., "Zero reconstructed centered moments from lossy raw diagnostics");
    const float next=std::nextafter(.5f, 1.f);
    const auto adjacent=measurement({.5f, next}).decode();
    require(adjacent.valid(), "One-ULP scene contrast was treated as a flat epsilon band");
    close(adjacent.zero_target(), (double(.5f)+next)*.5, 0., "One-ULP target lost its true midpoint");
  }

  void centered_temporal_controls_retain_lifecycle_and_rate() {
    const auto first=measurement({0.f, 0.f, 0.f, 1.f}).decode();
    const auto second=measurement({0.f, .25f, .25f, 1.f}).decode();
    fixture value; value.start(first);
    value.capture(second, 1800);
    const double alpha=-std::expm1(-.05/sunshine_camera_scene::time_constant_seconds);
    close(value.value.zero(), .5+alpha*(.375-.5), 4e-8, "Changed distribution bypassed zero EMA");
    require(value.value.value()==1.f, "Zero tracking modified gain despite unchanged maximum");
    const float held_zero=value.value.zero(), held_gain=value.value.value();
    auto invalid=second; invalid.mean_contrast_square=-1.;
    value.capture(invalid, 1850);
    require(!value.value.has_target() && value.value.zero()==held_zero && value.value.value()==held_gain,
      "Invalid centered evidence moved or retained a target");
    value.capture(second, 9000);
    require(value.value.zero()==held_zero && value.value.value()==held_gain,
      "Centered recovery banked unavailable elapsed time");
    float previous=value.value.zero();
    for (std::uint64_t now=9050; now<=11000; now+=50) {
      value.capture(second, now);
      require(std::abs(double(value.value.zero())-previous)*value.value.value()<=.05+2e-7,
        "Distribution-driven zero motion bypassed its display-budget rate bound");
      previous=value.value.zero();
    }
    const auto positive_flat=measurement({.75f, .75f}).decode();
    require(positive_flat.valid(), "Exact positive-flat centered moments were rejected");
    value.capture(positive_flat, 11250);
    require(value.value.has_target() && !value.value.has_gain_target() && value.value.value()==held_gain,
      "Positive flat target changed held gain");
    close(value.value.target_zero(), .75, 0., "Positive-flat zero was lost");
    const auto zero=measurement({0.f, 0.f}).decode();
    const float before=value.value.zero();
    value.capture(zero, 11500);
    require(!value.value.has_target() && value.value.zero()==before && value.value.value()==held_gain,
      "All-zero evidence fabricated a gain or zero target");
  }

  void raw_and_projection_callers_keep_centered_distribution() {
    namespace raw=sunshine_raw_scene;
    namespace projection=sunshine_projection_depth;
    for (const auto direction: {raw::orientation::normal, raw::orientation::reversed}) {
      const bool normal=direction==raw::orientation::normal;
      const measurement source(normal ? std::vector<float>{1.f,.75f,.75f,0.f} :
        std::vector<float>{0.f,.25f,.25f,1.f}, normal?1.f:0.f, normal?-1.f:1.f);
      raw::selected_frame current;
      current.basis_epoch=7; current.layout_epoch=3;
      current.source={0x100,9,0,1920,1080,0,0,1920,1080};
      current.direction=direction; current.depth_ready=current.aligned_viewport_assumed=true;
      raw::policy raw_policy;
      require(raw_policy.reset(current.basis_epoch, 1000), "Raw centered fixture reset rejected");
      raw_policy.configure({.5,1.,1.});
      const auto coefficients=projection::make(normal?1.:0., normal?-1.:1.);
      require(coefficients.valid(), "Projection centered fixture coefficients invalid");
      const projection::domain domain{3,7};
      projection::controller projected; projected.reset(domain,1000); projected.configure({.5,1.,1.});
      raw::output raw_out;
      projection::center_output projected_out;
      for (unsigned i=0;i!=5;++i) {
        const auto now=1000+i*250;
        current.frame={1+i,17};
        raw::sample packet;
        packet.id=1+i; packet.capture_ms=now; packet.metadata=current;
        packet.readback_frame=current.frame; packet.readback_source=current.source;
        packet.readback_layout_epoch=current.layout_epoch;
        packet.range_supplied=packet.range_valid=true;
        packet.range_min=source.raw_min; packet.range_max=source.raw_max;
        packet.moments=source.statistics;
        if (i==4) packet.moments.center=.25; // Wrong center must fail both callers.
        raw_out=raw_policy.update(current,&packet,now);
        projection::sample camera;
        camera.id=packet.id; camera.capture_ms=now; camera.logical_domain=domain;
        camera.projection=coefficients; camera.range_supplied=camera.range_valid=true;
        camera.range_min=packet.range_min; camera.range_max=packet.range_max; camera.moments=packet.moments;
        const auto accepted=projected.observe(camera,now);
        projected_out=projected.evaluate(domain,coefficients,now);
        if (i==3) {
          require(raw_out.ready && projected_out.ready && raw_out.depth_statistics.has_centered_moments &&
            projected_out.depth_statistics.has_centered_moments, "Controller caller lost centered metadata");
          close(raw_out.t0,.375,0.,"Raw caller used legacy midpoint");
          close(projected_out.q0,.375,0.,"Projection caller used legacy midpoint");
          close(raw_out.ui_midpoint_q,.5,0.,"Raw caller lost applied UI midpoint");
          close(projected_out.ui_midpoint_q,.5,0.,"Projection caller lost applied UI midpoint");
          close(raw_out.target_ui_midpoint_q,.5,0.,"Raw caller lost target UI midpoint");
          close(projected_out.target_ui_midpoint_q,.5,0.,"Projection caller lost target UI midpoint");
          require(raw_out.H==projected_out.K && raw_out.H==1.f, "Raw/projection gain diverged");
        }
        if (i==4) {
          require(raw_out.reason==raw::status::invalid_depth &&
            accepted==projection::center_status::invalid_depth && !raw_out.has_depth_statistics &&
            !projected_out.has_depth_statistics, "Bad centered payload survived a production caller");
          require(raw_out.ui_midpoint_q==.5f && projected_out.ui_midpoint_q==.5f &&
            raw_out.target_ui_midpoint_q==0. && projected_out.target_ui_midpoint_q==0.,
            "Invalid controller evidence erased applied UI midpoint or retained its target");
        }
      }
    }
  }

  void ui_midpoint_reproduces_legacy_motion_without_changing_scene_gain() {
    sunshine_scene_gain::policy value, legacy;
    const sunshine_scene_gain::limits budget{.5,1.,1.};
    value.configure(budget); legacy.configure(budget);
    const auto compare = [&] {
      // Range-only zero is the pre-contrast midpoint implementation. Running
      // it independently verifies the UI's entire admitted/held trajectory.
      require(value.initialized()==legacy.initialized() && value.samples()==legacy.samples() &&
        value.has_target()==legacy.has_target() && value.has_gain_target()==legacy.has_gain_target() &&
        value.value()==legacy.value() && value.target()==legacy.target(),
        "Independent UI tracking changed gain, calibration or evidence admission");
      require(value.ui_midpoint()==legacy.zero() && value.target_ui_midpoint()==legacy.target_zero(),
        "UI midpoint diverged from the former scene midpoint trajectory");
      require(value.ui_midpoint()>=value.zero() && value.target_ui_midpoint()>=value.target_zero(),
        "UI midpoint crossed behind the contrast zero");
    };
    const auto capture = [&](const depth_range &range, std::uint64_t now,
        sunshine_scene_feedback::sample feedback = {}) {
      auto old_range=range;
      if (!range.valid()) old_range={std::numeric_limits<double>::quiet_NaN(),0.};
      else old_range.has_centered_moments=false;
      value.advance(now); legacy.advance(now);
      value.synchronize(feedback); legacy.synchronize(feedback);
      value.observe(range,now,feedback); legacy.observe(old_range,now,feedback);
      value.update(now); legacy.update(now); compare();
    };
    const auto state = [&] { return std::array<float,3>{value.value(),value.zero(),value.ui_midpoint()}; };
    const std::array<depth_range,4> startup{
      measurement({0.f,.25f,.25f,1.f}).decode(),
      measurement({0.f,.125f,.125f,.5f}).decode(),
      measurement({.125f,.25f,.25f,.625f}).decode(),
      measurement({.25f,.375f,.375f,.75f}).decode()};
    compare();
    for (unsigned i=0;i!=startup.size();++i) {
      capture(startup[i],1000+i*250);
      require(value.initialized()==(i==3), "UI midpoint bypassed the shared four-sample startup");
    }
    close(value.zero(),.328125,0.,"Varying startup changed the existing contrast seed");
    close(value.ui_midpoint(),.40625,0.,"UI startup used only the latest midpoint instead of four-sample history");
    close(value.target_ui_midpoint(),.5,0.,"UI startup target did not preserve the latest admitted midpoint");

    const auto wide=measurement({0.f,.25f,.25f,1.f},0.f,64.f).decode();
    const auto shifted=measurement({.5f,.625f,.625f,1.f}).decode();
    for (unsigned i=0;i!=80;++i) {
      const auto previous=state();
      const auto &range=i%3==0 ? wide : i%3==1 ? shifted : startup[i%startup.size()];
      capture(range,1800+i*50);
      require(std::abs(double(value.ui_midpoint())-previous[2])*value.value()<=.05+2e-6,
        "UI midpoint exceeded the shared current-gain display-speed bound");
      require(std::abs(double(value.zero())-previous[1])*value.value()<=.05+2e-6,
        "Adding UI midpoint changed the contrast-zero movement bound");
    }
    auto held=state();
    capture(wide,5800,{1,true});
    require(state()==held && !value.has_target(), "Feedback cut moved either plane instead of holding both");
    capture(wide,5850,{1,false});
    require(state()==held, "Post-cut UI recovery banked elapsed time");
    capture(wide,5900,{1,false});
    auto invalid=shifted; invalid.mean_contrast_square=-1.; held=state();
    capture(invalid,5950,{1,false});
    require(state()==held && !value.has_target(), "Invalid moments moved the UI plane or retained its target");
    capture(shifted,6000,{1,false});
    require(state()==held, "Invalid evidence armed a catch-up step");

    auto zero_strength=budget; zero_strength.strength=0.;
    value.configure(zero_strength); legacy.configure(zero_strength);
    for (const auto now: {6050ULL,6300ULL,6550ULL}) {
      capture(wide,now,{1,false});
      require(state()==held, "Zero strength continued UI or scene adaptation");
    }
    value.configure(budget); legacy.configure(budget);
    capture(shifted,6600,{1,false});
    require(state()==held, "Strength resume granted hidden UI catch-up time");
    capture(shifted,6650,{1,false});
    held=state(); capture(wide,10000,{1,false});
    require(state()==held, "Presentation gap teleported the legacy UI midpoint");
    value.update(12000); legacy.update(12000); compare();
    require(state()==held && !value.has_target(), "Expired evidence kept moving the UI midpoint");
    capture(shifted,12500,{1,false});
    require(state()==held, "Expired target banked UI movement");

    const auto positive_flat=measurement({.75f,.75f}).decode();
    capture(positive_flat,12550,{1,false});
    require(value.has_target() && !value.has_gain_target() && value.value()==held[0] &&
      value.target_zero()==.75 && value.target_ui_midpoint()==.75,
      "Positive flat failed to converge both planes independently of held gain");
    held=state(); capture(measurement({0.f,0.f}).decode(),12600,{1,false});
    require(state()==held && !value.has_target(), "All-infinity depth fabricated a UI plane target");

    value.reset_reference(); legacy.reset_reference(); compare();
    require(value.ui_midpoint()==0.f && value.target_ui_midpoint()==0., "Reference reset retained UI midpoint history");
    for (unsigned i=0;i!=4;++i) capture(shifted,12750+i*250,{1,false});
    close(value.zero(),.6875,0.,"Reference reset changed contrast reseeding");
    close(value.ui_midpoint(),.75,0.,"Reference reset did not reseed independent UI midpoint");
  }
}

int main() {
  try {
    distribution_changes_zero_without_changing_gain();
    centered_payload_corruption_never_falls_back();
    near_flat_uses_centered_evidence_across_units();
    centered_temporal_controls_retain_lifecycle_and_rate();
    raw_and_projection_callers_keep_centered_distribution();
    ui_midpoint_reproduces_legacy_motion_without_changing_scene_gain();
    std::puts("PASS contrast zero: full distributions, independent gain/UI midpoint, malformed payload rejection, near-flat/extreme units, temporal lifecycle and raw/projection callers");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr,"FAIL contrast zero: %s\n",error.what());
    return 1;
  }
}

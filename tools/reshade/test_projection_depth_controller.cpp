// SPDX-License-Identifier: GPL-3.0-only
#include "projection_depth_controller.h"
#include "raw_scene_policy.h"

#include <cstdio>
#include <stdexcept>

namespace {
  using namespace sunshine_projection_depth;
  void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
  void close(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance) {
      std::fprintf(stderr,"%s: actual=%.12g expected=%.12g\n",message,actual,expected);
      throw std::runtime_error(message);
    }
  }
  constexpr domain viewport{3,7};
  sample plane(std::uint64_t id, std::uint64_t capture_ms, float q,
      coefficients projection=make(0,.0625), domain source=viewport) {
    sample value;
    value.id=id; value.capture_ms=capture_ms; value.logical_domain=source; value.projection=projection;
    value.raw.fill(projection.shader_A+q/projection.inverseB);
    return value;
  }
  struct fixture {
    controller policy;
    coefficients projection{make(0,.0625)};
    center_output output;
    std::uint64_t id{};
    explicit fixture(coefficients value=make(0,.0625)) : projection(value) { policy.reset(viewport,1000); }
    center_output capture(float q,std::uint64_t tick) {
      policy.observe(plane(++id,tick,q,projection),tick);
      return output=policy.evaluate(viewport,projection,tick);
    }
    center_output tick(std::uint64_t time) { return output=policy.evaluate(viewport,projection,time); }
    center_output start(float q=.125f) {
      for (std::uint64_t now=1000;now<=1750;now+=250) {
        capture(q,now);
        require(output.ready==(now==1750),"Camera reference did not require four spaced center captures");
      }
      close(output.K,1.0/q,1e-5,"Reference distance is not inverse center depth");
      close(output.q0,q,1e-7,"Initial zero does not use the same bounded center reference");
      require(output.calibration_samples==4,"Initial sample accounting differs from raw fallback");
      return output;
    }
  };

  void screen_plane_controls_reference() {
    fixture a,b;
    const auto initial=a.start(.125f);b.start(.5f);
    for(std::uint64_t time=1800;time<=10000;time+=50) {
      const auto current=a.capture(.25f,time);b.capture(.25f,time);
      require(current.ready && current.K==1.f/current.q0,"Current scale does not follow published zero");
      require(current.q0>=initial.q0 && current.q0<=.25f,"Screen plane overshot its target");
      close(current.target_K,4,0,"UI scale target differs from inverse target zero");
    }
    close(a.output.q0,.25,1e-7,"Screen plane failed to converge");
    close(a.output.K,b.output.K,1e-5,"Initial scene permanently affected scale");
    auto other=domain{viewport.epoch,viewport.viewport+1};
    require(a.policy.observe(plane(1000,10100,.1f,a.projection,other),10100)==center_status::invalid_domain,"Other viewport supplied a target");
    require(!a.policy.evaluate(other,a.projection,10100).ready,"Other viewport inherited controls");
    require(a.tick(10100).ready,"Other viewport erased original screen plane");
  }

  void equal_relative_depth_has_equal_separation() {
    fixture value;value.start(.25f);
    std::uint64_t now=1800;
    for(const float center:{.01f,.75f,.25f}) {
      const auto end=now+12000;
      for(;now<=end;now+=50) {
        const auto next=value.capture(center,now);
        require(next.ready && next.K==1.f/next.q0,"Current reference detached from the screen plane");
        close(reference_zpd*next.K*(next.q0-next.q0*.5),reference_zpd*.5,1e-7,"Equal relative depth changed screen disparity");
      }
      close(value.output.q0,center,2e-7,"Room/wall transition retained initial scene");
    }
  }

  void raw_camera_parity() {
    namespace raw=sunshine_raw_scene;
    for (const bool reversed : {false,true}) for (const double units : {.01,1.0,100.0}) {
      const double B=(reversed?1:-1)*units;
      fixture camera(make(reversed?0:1,B));
      raw::policy fallback;
      require(fallback.reset(7,1000),"Raw parity reset failed");
      raw::selected_frame selected;
      selected.basis_epoch=7; selected.layout_epoch=3;
      selected.source={0x100,9,0,1920,1080,0,0,1920,1080};
      selected.direction=reversed?raw::orientation::reversed:raw::orientation::normal;
      selected.depth_ready=selected.aligned_viewport_assumed=true;
      bool saw_ready=false;
      for (std::uint64_t tick=1000;tick<=12000;tick+=250) {
        const float t=tick<2000?.00175f:tick<7000?.0005f:.01f;
        const float stored=reversed?t:1.f-t;
        float q{};
        require(inverse_distance(camera.projection,stored,q),"Parity fixture conversion failed");
        const auto actual=camera.capture(q,tick);
        selected.frame={tick,17};
        raw::sample packet;
        packet.id=packet.capture_ms=tick; packet.metadata=selected;
        packet.readback_frame=selected.frame; packet.readback_source=selected.source;
        packet.readback_layout_epoch=selected.layout_epoch;
        packet.raw.fill(stored);
        const auto expected=fallback.update(selected,&packet,tick);
        require(actual.ready==expected.ready,"Camera/raw readiness differs for same valid source samples");
        if (!actual.ready) continue;
        require(actual.K==1.f/actual.q0 && expected.H==1.f/expected.t0,"Camera/raw reference is not current zero");
        saw_ready=true;
        close(actual.K*camera.projection.inverseB*(reversed?1:-1),expected.H,expected.H*2e-6,
          "Camera/raw effective depth gain differs for same infinite-perspective pixels");
        const float sample_q=q*1.5f;
        const double camera_field=reference_zpd*actual.K*(actual.q0-sample_q);
        const double raw_sample=sample_q*units;
        close(camera_field,reference_zpd*expected.H*(expected.t0-raw_sample),2e-6,
          "Camera/raw paths produce different disparity at equal user strength");
      }
      require(saw_ready,"Camera/raw parity never initialized a screen plane");
    }
  }

  void units_storage_and_clipping() {
    // Physical q changes with units, but K*q and K*(q0-q) must not.
    for (double unit : {.001,1.0,100.0,1e6}) {
      fixture baseline,changed(make(0,.0625*unit,2,-.5));
      for (std::uint64_t time=1000;time<=6000;time+=250) {
        const float q=time<2000?.125f:.25f;
        const auto a=baseline.capture(q,time), b=changed.capture(float(q/unit),time);
        require(a.ready==b.ready,"Unit/precision change altered calibration readiness");
        if (!a.ready) continue;
        close(a.K*q,b.K*(q/unit),2e-5,"Unit/precision change altered prepared depth");
        close(reference_zpd*a.K*(a.q0-.375),reference_zpd*b.K*(b.q0-.375/unit),2e-6,
          "Unit/precision change altered actual disparity field");
      }
    }
    fixture value;
    const auto original=value.start();
    value.projection=make(0,.125); // Different near, same game units and physical scene.
    const auto next=value.capture(.125f,2000);
    require(next.K==original.K && next.q0==original.q0,"Clipping change changed strength or physical screen plane");
    float q{};
    require(inverse_distance(value.projection,.0625f,q) && q==.5f,
      "Current pixels used smoothed or stale projection coefficients");
    // Finite-far decoding stays physical. Unlike raw fallback, the camera knows
    // the far offset; identical normalization is not asserted for unknown encodings.
    value.projection=make(-.125/(100-.125),.125*100/(100-.125));
    const auto finite=value.capture(.125f,2250);
    close(finite.K,original.K,2e-6,"Finite far implicitly changed artistic scale");
    close(finite.q0,original.q0,2e-7,"Finite far changed physical zero");
  }

  void freshness_and_gaps_suspend_motion() {
    fixture value;value.start();
    value.capture(.5f,2000);
    const auto before=value.output;
    value.tick(3000);
    require(value.output.q0==before.q0,"Long presentation gap banked movement");
    value.capture(.5f,5000);
    require(value.output.q0==before.q0,"Expired-target return banked movement");
    const auto resumed=value.capture(.5f,5050);
    require(resumed.q0>before.q0 && resumed.K<before.K,"Fresh continuity did not resume coupled zero/reference");
    require(value.policy.observe(plane(value.id,5050,1.f),5100)==center_status::stale_sample,"Repeated capture supplied fresh evidence");
    const auto held=value.tick(7000);
    require(held.ready && held.reason==center_status::holding,"Target expiry discarded usable current depth");
    require(value.tick(7050).q0==held.q0,"Expired evidence continued moving the plane");
  }

  void explicit_reference_reset_rejects_pre_action_evidence() {
    fixture value;
    const auto initial=value.start();
    require(!value.policy.reset_reference({viewport.epoch,viewport.viewport+1},2000),
      "Wrong viewport reset the established reference");
    require(value.tick(2000).K==initial.K,"Rejected reset changed the reference");
    require(value.policy.reset_reference(viewport,2000),"Explicit reference reset failed");
    const auto pending=value.tick(2000);
    require(!pending.ready && pending.K==0 && pending.calibration_samples==0,
      "Explicit reset retained the old reference or zero");
    require(value.policy.observe(plane(100,2000,.5f),2000)==center_status::stale_sample,
      "Same-tick pending evidence seeded the new reference");
    require(value.policy.observe(plane(value.id,2001,.5f),2001)==center_status::stale_sample,
      "Already consumed capture ID seeded the new reference");
    for (std::uint64_t tick=2001;tick<=2751;tick+=250) {
      const auto next=value.capture(.5f,tick);
      require(next.ready==(tick==2751),"Reference reset bypassed fresh four-sample startup");
    }
    require(value.output.K==2.f && value.output.q0==.5f,
      "Explicit reset failed to replace reference and zero together");
  }

  void explicit_reset_during_new_domain_adoption_is_strict() {
    fixture value;
    value.start();
    const domain replacement{viewport.epoch+1,viewport.viewport+1};
    // The exporter adopts a new projection domain before applying a pending
    // explicit action. Ordinary adoption alone permits a same-tick first sample;
    // the explicit reset must tighten that boundary for this newly adopted domain.
    value.policy.reset(replacement,2000);
    require(value.policy.reset_reference(replacement,2000),
      "Explicit action failed after adopting a new projection domain");
    require(value.policy.observe(plane(100,2000,.9f,value.projection,replacement),2000)==center_status::stale_sample,
      "New-domain same-timestamp pre-action evidence seeded the reference");
    require(value.policy.observe(plane(101,2001,.9f,value.projection,viewport),2001)==center_status::invalid_domain,
      "Old projection domain seeded the reset reference");
    for (std::uint64_t index=0;index<4;++index) {
      const auto tick=2001+250*index;
      value.policy.observe(plane(index+1,tick,.5f,value.projection,replacement),tick);
      const auto result=value.policy.evaluate(replacement,value.projection,tick);
      require(result.calibration_samples==index+1 && result.ready==(index==3),
        "New-domain reset did not require exactly four fresh post-action captures");
      if (index==3) require(result.K==2.f && result.q0==.5f,
        "New-domain reset reference contains pre-action depth evidence");
    }
  }

  void center_clears_stale_evidence_and_domains() {
    fixture value;
    auto initial=value.start(.5f);
    auto packet=plane(++value.id,2000,.5f);
    packet.raw.fill(0.f);
    packet.raw[7*grid_width+14]=.03125f; // One valid physical center q=.5.
    packet.raw[7*grid_width+15]=1.f;
    packet.raw[8*grid_width+14]=NAN;
    packet.raw[8*grid_width+15]=-1.f;
    require(value.policy.observe(packet,2000)==center_status::ready,"Valid center rejected with ambiguous neighbors");
    require(value.tick(2000).K==initial.K,"Filtered ambiguous cells biased gain");
    packet.id=++value.id; packet.capture_ms=2250; packet.raw.fill(1.f);
    require(value.policy.observe(packet,2250)==center_status::invalid_depth,"All-clear scene supplied reference");
    const auto held=value.tick(2250);
    require(held.ready && held.reason==center_status::holding && held.K==initial.K && held.q0==initial.q0,
      "All-clear measurement discarded valid depth controls");
    require(!value.policy.evaluate(viewport,{},2300).ready,"Missing projection authorized rendering");
    require(!value.tick(2200).ready,"Backward wall clock admitted");
    require(value.tick(2500).K==initial.K,"Clock recovery accumulated strength");
    require(value.policy.observe(plane(++value.id,2300,.75f),5000)==center_status::stale_sample,
      "Arrival timestamp refreshed stale capture");
    value.policy.reset({viewport.epoch+1,viewport.viewport},5100);
    const auto reset=value.policy.evaluate({viewport.epoch+1,viewport.viewport},value.projection,5100);
    require(!reset.ready && reset.K==0 && reset.calibration_samples==0,"New unit/source domain inherited strength");

    fixture expired;
    expired.capture(.125f,1000);
    expired.tick(2500);
    expired.capture(.25f,2750);
    require(!expired.output.ready && expired.output.calibration_samples==1,
      "Expired startup reference survived into a new scene");
  }

  void unsupported_rendering_still_recovers_numerically() {
    fixture value;
    const auto initial=value.start(.0078125f);
    value.projection=make(0,1e-38); // Valid coefficients, but .05*128*1e38 overflows FP32.
    require(value.projection.valid() && !supports_scale(value.projection,initial.K),"Fixture does not exercise shader range");
    const auto unsafe=value.tick(2000);
    require(!unsafe.ready && unsafe.reason==center_status::unsupported_shader_domain,"Unsupported encoding rendered");
    bool recovered=false;
    for(std::uint64_t time=2050;time<=8000;time+=50) {
      const auto next=value.capture(2.f,time);
      require(next.K==1.f/next.q0,"Unready numerical state detached gain and zero");
      recovered|=next.ready;
    }
    require(recovered && value.output.ready && value.output.q0>1.9f,"Old unsupported scale blocked numerical recovery");
    fixture far;
    for(std::uint64_t time=1000;time<=1750;time+=250)far.capture(.00001f,time);
    require(far.output.ready && far.output.initialized && far.output.K>99999.f,
      "Finite large scale retained the old reciprocal FP16 restriction");
    for(std::uint64_t time=1800;time<=15000;time+=50)far.capture(.125f,time);
    require(far.output.ready,"Representable large scale failed ordinary scene adaptation");
  }

}
int main() {
  try {
    screen_plane_controls_reference();
    equal_relative_depth_has_equal_separation();
    raw_camera_parity();
    units_storage_and_clipping();
    freshness_and_gaps_suspend_motion();
    explicit_reference_reset_rejects_pre_action_evidence();
    explicit_reset_during_new_domain_adoption_is_strict();
    center_clears_stale_evidence_and_domains();
    unsupported_rendering_still_recovers_numerically();
    std::puts("PASS projection controls: current screen-plane reference, raw/unit parity, explicit reset, smooth recovery and encoding domains");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr,"FAIL projection controls: %s\n",error.what());
    return 1;
  }
}

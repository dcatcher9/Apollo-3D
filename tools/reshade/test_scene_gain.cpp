// SPDX-License-Identifier: GPL-3.0-only
#include "scene_gain.h"
#include "provided_raw_scene.h"
#include "projection_depth_controller.h"
#include <cstdio>
#include <stdexcept>

namespace {
  void require(bool value,const char *message) { if(!value) throw std::runtime_error(message); }
  using feedback=sunshine_scene_feedback::sample;
  struct fixture {
    sunshine_scene_gain::policy gain;
    feedback context{1};
    void capture(double zero,std::uint64_t time) {
      gain.advance(time);gain.synchronize(context);gain.observe(zero,time,context);gain.update(time);
      if(gain.initialized()) require(gain.value()==1.f/gain.zero(),"Gain does not use the published float screen plane");
    }
    void start(double zero=.25) {
      for(std::uint64_t time=1000;time<=1750;time+=250) capture(zero,time);
      require(gain.initialized(),"Initial screen plane failed");
    }
  };
  void screen_plane_forgets_initial_scene_and_preserves_units() {
    fixture a,b,units;
    a.start(.25);b.start(.5);units.start(.0025);
    float previous=a.gain.zero();
    for(std::uint64_t time=1800;time<=11750;time+=50) {
      a.capture(.375,time);b.capture(.375,time);units.capture(.00375,time);
      require(a.gain.zero()>=previous && a.gain.zero()<=.375f,"Screen plane overshot or reversed its target");
      require(std::abs(a.gain.zero()-100*units.gain.zero())<1e-6,"Changing game units changed the screen-plane trajectory");
      require(std::abs(a.gain.value()*.15-units.gain.value()*.0015)<1e-6,"Equal relative depth produced different disparity");
      previous=a.gain.zero();
    }
    require(std::abs(a.gain.zero()-.375)<1e-7 && a.gain.zero()==b.gain.zero(),"Initial scene permanently biased the screen plane");
    require(a.gain.target()==1.f/.375f,"UI target is not inverse target zero");
  }
  void startup_spacing_reset_and_no_banked_time() {
    fixture a;
    // Synchronize revision before admitted startup observations.
    a.gain.synchronize(a.context);
    for(auto time:{1000ULL,1000ULL,999ULL,1100ULL}) a.gain.observe(.25,time,a.context);
    require(a.gain.samples()==1,"Repeated/backward/close captures qualified startup");
    for(auto time:{1250ULL,1500ULL,1750ULL}) a.capture(.25,time);
    require(a.gain.initialized(),"Fresh spaced startup failed");
    a.capture(.5,2000);const auto held=a.gain.zero();
    a.capture(.5,5000);
    require(a.gain.zero()==held,"Gap banked screen-plane movement");
    a.capture(.5,5250);
    require(a.gain.zero()>held,"Fresh continuity did not resume smoothing");
    const auto before=a.gain.zero();const auto prior=a.context;
    ++a.context.revision;a.gain.synchronize(a.context);
    a.gain.observe(.01,5300,prior);a.gain.update(5300);
    require(a.gain.zero()==before && a.gain.target()==0,"Old revision revived stale target");
    a.capture(.125,5400);
    require(a.gain.zero()==before,"First post-reset capture banked stale time");
    a.gain.reset_reference();
    require(!a.gain.initialized() && !a.gain.accepts(prior),"Explicit reset retained zero or lost revision watermark");
    for(auto time:{5500ULL,5750ULL,6000ULL,6250ULL}) a.capture(.125,time);
    require(a.gain.value()==8 && a.gain.zero()==.125f,"Explicit reset failed to choose fresh plane");
  }
  void positive_relative_speed_and_expiry() {
    fixture a;a.start(.001);
    float previous=a.gain.zero();
    for(std::uint64_t time=1800;time<=2300;time+=50) {
      a.capture(1000,time);
      require(a.gain.zero()>previous && a.gain.zero()<=previous*1.100001f,"Screen plane exceeded relative speed bound");
      previous=a.gain.zero();
    }
    for(std::uint64_t time=2350;time<=3300;time+=50) {
      a.capture(1e-20,time);
      require(a.gain.zero()>0 && a.gain.zero()>=previous*.899999f,"Downward step crossed zero or exceeded speed bound");
      previous=a.gain.zero();
    }
    a.gain.update(5000);const auto held=a.gain.zero();a.gain.update(5050);
    require(a.gain.initialized() && !a.gain.has_target() && a.gain.zero()==held,"Expired evidence moved or erased numerical state");
  }
  void controllers_reject_prior_history_without_losing_depth() {
    sunshine_raw_scene::policy raw;
    raw.reset(1,1000);
    sunshine_projection_depth::controller camera;
    const sunshine_projection_depth::domain domain{1,0};
    const auto projection=sunshine_projection_depth::make(0,1);
    camera.reset(domain,1000);
    sunshine_scene_depth::frame source;
    source.epoch=1; source.source_id=1;
    source.projection.reversed=source.projection.direction_supplied=true;
    source.feedback.revision=1;
    std::array<float,32*18> pixels; pixels.fill(.25f);
    for (auto time=1000;time<=1750;time+=250) {
      source.tick=time; source.sequence++;
      const auto current=sunshine_provided_raw::selected(source,1,true);
      const auto packet=sunshine_provided_raw::measured(source,source.sequence,1,pixels);
      raw.update(current,&packet,time);
      camera.synchronize_feedback(source.feedback);
      sunshine_projection_depth::sample sample{source.sequence,source.tick,domain,projection,pixels,source.feedback};
      camera.observe(sample,time); camera.evaluate(domain,projection,time);
    }
    // Reset during a zero-plane transition must hold the applied zero, not keep
    // following an obsolete target from the previous scene.
    source.tick=1900; source.sequence++;
    pixels.fill(.5f);
    auto moving_current=sunshine_provided_raw::selected(source,1,true);
    auto moving=sunshine_provided_raw::measured(source,source.sequence,1,pixels);
    raw.update(moving_current,&moving,1900);
    sunshine_projection_depth::sample moving_camera{source.sequence,source.tick,domain,projection,pixels,source.feedback};
    camera.observe(moving_camera,1900); camera.evaluate(domain,projection,1900);
    const auto raw_before=raw.evaluate(moving_current,1950);
    const auto camera_before=camera.evaluate(domain,projection,1950);
    require(raw_before.t0>.25f && camera_before.q0>.25f, "Reset transition fixture did not move zero");
    source.tick=2000; source.sequence++;
    auto old=sunshine_provided_raw::measured(source,source.sequence,1,pixels);
    old.raw.fill(.01f);
    source.feedback.revision=2;
    const auto current=sunshine_provided_raw::selected(source,1,true);
    raw.bind(current,2000);
    raw.observe(old,current.frame,2000);
    auto result=raw.evaluate(current,2000);
    require(result.ready && result.H==raw_before.H && result.t0==raw_before.t0, "Raw reset lost stereo or followed old target");
    require(result.target_H==0, "Raw reset exposed its old scale target to the UI");
    camera.synchronize_feedback(source.feedback);
    sunshine_projection_depth::sample prior{source.sequence,source.tick,domain,projection,old.raw,old.metadata.feedback};
    require(camera.observe(prior,2000)==sunshine_projection_depth::center_status::stale_sample,
      "Camera accepted stale history");
    const auto output=camera.evaluate(domain,projection,2000);
    require(output.ready && output.K==camera_before.K && output.q0==camera_before.q0, "Camera reset discarded stereo or followed old target");
    require(output.target_K==0, "Camera reset exposed its old scale target to the UI");
    source.tick=2250; source.sequence++;
    source.feedback.reset=true;
    pixels.fill(0.f);
    const auto reset_current=sunshine_provided_raw::selected(source,1,true);
    const auto reset_packet=sunshine_provided_raw::measured(source,source.sequence,1,pixels);
    const auto reset_output=raw.update(reset_current,&reset_packet,2250);
    require(reset_output.ready && reset_output.H==raw_before.H && reset_output.t0==raw_before.t0,
      "Clear center on reset frame discarded established raw stereo");
  }
}
int main() {
  try {
    screen_plane_forgets_initial_scene_and_preserves_units();
    startup_spacing_reset_and_no_banked_time();
    positive_relative_speed_and_expiry();
    controllers_reject_prior_history_without_losing_depth();
    std::puts("PASS shared screen plane: reciprocal float identity, units, convergence, positive speed, reset and history safety");
    return 0;
  } catch(const std::exception &e) {std::fprintf(stderr,"FAIL screen plane: %s\n",e.what());return 1;}
}

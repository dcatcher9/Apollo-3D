// SPDX-License-Identifier: GPL-3.0-only
// Current zero-plane normalization driving the actual, separately supplied shader.
// Registration and projection inputs are synthetic fixture-owned facts.
#pragma once
#include "projection_depth_controller.h"
#include "raw_scene_policy.h"

namespace sunshine_stereo_reference_fixture {
  namespace camera = sunshine_projection_depth;
  namespace raw = sunshine_raw_scene;
  struct controls {
    float A{}, inverseB{}, reference{}, zero{}, budget{camera::reference_zpd};
    float minimum{}, maximum{1};
    int basis{};
  };
  inline controls parameters(const camera::center_output &state, const camera::coefficients &encoding) {
    return {encoding.shader_A,encoding.inverseB,state.K,state.q0,camera::reference_zpd,
      encoding.raw_min,encoding.raw_max,0};
  }
  inline controls parameters(const raw::output &state) {
    return {state.shader_A,state.shader_inverseB,state.H,state.t0,state.referenceZPD,0,1,1};
  }
  struct camera_source {
    camera::controller policy;
    camera::coefficients encoding;
    camera::center_output state;
    camera::domain domain{1,7};
    std::uint64_t id{}, tick{1000};
    explicit camera_source(camera::coefficients value):encoding(value) { policy.reset(domain,tick); }
    void capture(float stored) {
      camera::sample input;
      input.id=++id; input.capture_ms=tick; input.logical_domain=domain;
      input.projection=encoding; input.raw.fill(stored);
      policy.observe(input,tick);
      state=policy.evaluate(domain,encoding,tick);
      tick+=250;
    }
    void initialize(float stored) {
      for (unsigned i=0;i<4;++i) {
        capture(stored);
        sunshine_parity::need(state.ready==(i==3),"Production camera reference startup differs from four fresh captures");
      }
    }
  };
  struct raw_source {
    raw::policy policy;
    raw::selected_frame selected;
    raw::output state;
    std::uint64_t id{}, tick{1000};
    raw_source(unsigned width,unsigned height,bool reversed) {
      sunshine_parity::need(policy.reset(1,tick),"Raw fixture epoch was rejected");
      selected.basis_epoch=1; selected.layout_epoch=1;
      selected.source={0x1234,1,0,width,height,0,0,width,height};
      selected.direction=reversed?raw::orientation::reversed:raw::orientation::normal;
      selected.depth_ready=selected.aligned_viewport_assumed=true;
    }
    void capture(float stored) {
      selected.frame={++id,1};
      raw::sample input;
      input.id=id; input.capture_ms=tick; input.metadata=selected;
      input.readback_frame=selected.frame; input.readback_source=selected.source;
      input.readback_layout_epoch=selected.layout_epoch; input.raw.fill(stored);
      state=policy.update(selected,&input,tick);
      tick+=250;
    }
    void initialize(float stored) {
      for (unsigned i=0;i<4;++i) {
        capture(stored);
        sunshine_parity::need(state.ready==(i==3),"Production raw reference startup differs from four fresh captures");
      }
    }
  };

  template<class Fixture,class Plane,class Capture,class Metadata,class Ready>
  void run(Fixture &f,unsigned width,unsigned height,Plane plane,Capture capture,
      Metadata metadata,Ready ready,reshade::api::effect_runtime *runtime,const std::filesystem::path &directory) {
    using sunshine_parity::need;
    namespace fs=std::filesystem;
    const auto output=directory/"stereo-reference";
    need(!fs::exists(output),"Current-zero GPU evidence requires a fresh directory");
    fs::create_directories(output);
    std::ofstream log(output/"measurements.csv");
    log<<std::setprecision(12)<<"case,strength,reference,zero,raw,left_px,right_px,disparity_px,prepared_depth,oracle_q_ratio,oracle_disparity_px,oracle_prepared_depth\n";
    std::ofstream scope(output/"scope.txt");
    scope<<"Actual production camera/raw policies normalize by their current smoothed zero plane. "
      <<"Independent oracle: field=0.05*(1-q/q0)*strength/100; pair disparity=2*height/2160*100*field. "
      <<"Known synthetic depth/registration and projection coefficients. "
      <<"No live pose discovery, SDK integration, performance result or game-quality claim. "
      <<"Shift measurements use actual exported texture alignment, not a replica of the renderer.\n";
    const float rectangle[]{0,0,1,1};
    runtime->set_uniform_value_float(f.uniform("Sunshine_CameraDepthRect"),rectangle,4);
    f.set_float("Sunshine_CameraStrengthBlend",1);
    const auto apply=[&](const controls &value) {
      metadata(value.A,value.inverseB,value.reference,value.budget,value.zero);
      const float range[]{value.minimum,value.maximum};
      runtime->set_uniform_value_float(f.uniform("Sunshine_CameraRawDepthRange"),range,2);
      f.set_int("Sunshine_CameraCoordinateBasis",value.basis);
      ready(true);
    };
    camera_source baseline(camera::make(0,.0625));
    baseline.initialize(.0078125f); // Current zero at distance 8, inverse distance 1/8.
    const auto base=parameters(baseline.state,baseline.encoding);
    apply(base); plane(.0078125f); f.set_float("Depth_Adjustment",0);
    const auto mono=capture();
    sunshine_parity::write_bytes(output/"mono.sbs",mono.data(),mono.size());
    struct rendered { std::vector<std::uint8_t> pixels; float disparity{},green{}; };
    const auto render=[&](const controls &value,float stored,float strength,double oracle_ratio,const std::string &label) {
      need(std::isfinite(value.zero) && value.zero>0 &&
        std::abs(double(value.reference)*value.zero-1)<1e-5,
        "Ready normalization is not the reciprocal of the current smoothed zero plane");
      apply(value); plane(stored); f.set_float("Depth_Adjustment",strength);
      rendered image;
      image.pixels=capture();
      const auto shifts=f.measure_eye_shifts(mono,image.pixels,label.c_str());
      image.disparity=shifts[0]-shifts[1];
      const auto prepared=f.read(f.linear_depth.p);
      const auto desc=f.linear_depth->GetDesc();
      const auto pixel=std::size_t(desc.Height/2)*desc.Width+desc.Width/2;
      std::uint16_t channels[2]{};
      std::memcpy(channels,prepared.data()+pixel*4,4);
      image.green=sunshine_parity::decode_half(channels[1]);
      // Expected geometry comes from the fixture's known world distances and
      // settled target, never from the controller's emitted gain or decoder.
      const double expected=1/(1+oracle_ratio);
      const double expected_disparity=2*(double(height)/2160*100)*.05*(1-oracle_ratio)*(strength/100);
      log<<label<<','<<strength<<','<<value.reference<<','<<value.zero<<','<<stored<<','
        <<shifts[0]<<','<<shifts[1]<<','<<image.disparity<<','<<image.green<<','
        <<oracle_ratio<<','<<expected_disparity<<','<<expected<<'\n';
      log.flush();
      need(log.good(),"Cannot write current-zero measurements");
      sunshine_parity::write_bytes(output/(label+".sbs"),image.pixels.data(),image.pixels.size());
      need(std::isfinite(image.green) && std::abs(image.green-expected)<.002,
        "Actual prepared depth differs from the independently known depth ratio");
      need(std::abs(image.disparity-expected_disparity)<=.5,
        "Actual binocular displacement differs from the independent current-zero geometry");
      return image;
    };
    const auto equivalent=[&](const rendered &expected,const rendered &actual,const char *message) {
      need(f.image_difference(expected.pixels,actual.pixels)<.01f &&
        std::abs(expected.disparity-actual.disparity)<=.5f,message);
    };
    const auto near_image=render(base,.015625f,100,2,"reference-near"); // Z=4, Z0=8.
    const auto far_image=render(base,.00390625f,100,.5,"reference-far"); // Z=16, Z0=8.
    const float original_span=std::abs(near_image.disparity-far_image.disparity);
    need(original_span>1,"Current-zero fixture did not establish visible stereo separation");
    const auto zero=render(base,.0078125f,100,1,"reference-zero");
    need(std::abs(zero.disparity)<=.5f,"The independently selected zero plane has nonzero binocular disparity");
    const auto half_near=render(base,.015625f,50,2,"strength-half-near");
    const auto half_far=render(base,.00390625f,50,.5,"strength-half-far");
    const float half_span=std::abs(half_near.disparity-half_far.disparity);
    need(half_span>.5f && original_span>half_span+.5f && std::abs(original_span-2*half_span)<=1.5f,
      "Strength slider failed to scale actual binocular separation");
    const auto no_strength=render(base,.015625f,0,2,"strength-zero");
    need(f.image_difference(mono,no_strength.pixels)<.003f && std::abs(no_strength.disparity)<=.5f,
      "Zero strength changed mono color or separated the eyes");

    for (const double unit:{.001,100.0}) {
      camera_source changed(camera::make(0,.0625*unit));
      changed.initialize(.0078125f);
      const auto value=parameters(changed.state,changed.encoding);
      need(std::abs(value.reference/unit-base.reference)<1e-4,
        "Production reference did not compensate camera world units");
      equivalent(near_image,render(value,.015625f,100,2,"units-"+std::to_string(unit)+"-near"),
        "World-unit changes altered exported near-eye images");
      equivalent(far_image,render(value,.00390625f,100,.5,"units-"+std::to_string(unit)+"-far"),
        "World-unit changes altered exported far-eye images");
    }
    camera_source normal(camera::make(1,-.0625));
    normal.initialize(1-.0078125f);
    equivalent(near_image,render(parameters(normal.state,normal.encoding),1-.015625f,100,2,"normal-z-near"),
      "Normal/reversed depth changed the same stereo geometry");
    camera_source packed(camera::make(0,.0625,2,-.5));
    packed.initialize((.0078125f+.5f)/2);
    equivalent(near_image,render(parameters(packed.state,packed.encoding),(.015625f+.5f)/2,100,2,"packed-depth-near"),
      "Affine raw encoding changed exported stereo geometry");
    for (bool reversed:{false,true}) {
      raw_source fallback(width,height,reversed);
      fallback.initialize(reversed?.0078125f:1-.0078125f);
      equivalent(near_image,render(parameters(fallback.state),reversed?.015625f:1-.015625f,100,2,
        reversed?"raw-reversed-near":"raw-normal-near"),
        "Equivalent projection and raw-reference paths disagree in exported geometry");
    }
    constexpr double n=.0625, fclip=1024;
    constexpr double A=-n/(fclip-n), B=n*fclip/(fclip-n);
    camera_source known(camera::make(A,B));
    known.initialize(float(A+B/8));
    render(parameters(known.state,known.encoding),float(A+B/4),100,2,"finite-far-near");
    render(parameters(known.state,known.encoding),float(A+B/32),100,.25,"finite-far-background");

    // The same two world surfaces intentionally change their pairwise strength
    // when the current zero moves. At settled Z0=4, their ratios are 1 and 1/4;
    // on returning to Z0=8, their ratios are 2 and 1/2 again.
    raw_source raw_history(width,height,true);
    raw_history.initialize(.0078125f);
    for (unsigned phase=0;phase<2;++phase) {
      const float center=phase==0?.015625f:.0078125f;
      float previous_zero=baseline.state.q0;
      for (unsigned sample=0;sample<24;++sample) {
        baseline.capture(center); raw_history.capture(center);
        need(baseline.state.ready && raw_history.state.ready,"A valid room/wall transition lost depth readiness");
        need(std::abs(double(baseline.state.K)*baseline.state.q0-1)<1e-5 &&
          std::abs(double(raw_history.state.H)*raw_history.state.t0-1)<1e-5,
          "Gain did not follow the current zero throughout a transition");
        need(phase==0?baseline.state.q0>=previous_zero:baseline.state.q0<=previous_zero,
          "Smoothed zero moved away from the constant new scene target");
        need(std::abs(baseline.state.q0-raw_history.state.t0*16)<1e-6,
          "Equivalent camera/raw current-zero transitions disagree");
        previous_zero=baseline.state.q0;
        if (sample==3) {
          need(baseline.state.q0>.125f && baseline.state.q0<.25f,
            "Current zero jumped directly to the target instead of smoothing");
          // Unlike settled scene ratios, a transient screen-plane probe is
          // deliberately placed at the published current zero itself.
          const auto value=parameters(baseline.state,baseline.encoding);
          render(value,value.A+value.zero/value.inverseB,100,1,
            phase==0?"wall-transition-zero":"return-transition-zero");
        }
      }
      const std::string label=phase==0?"wall":"room-return";
      const auto camera_value=parameters(baseline.state,baseline.encoding), raw_value=parameters(raw_history.state);
      const double near_ratio=phase==0?1:2, far_ratio=phase==0?.25:.5;
      need(std::abs(camera_value.zero-center*16)<1e-5,
        "Current-zero controller did not reach the independently supplied scene target");
      const auto camera_near=render(camera_value,.015625f,100,near_ratio,label+"-camera-near");
      const auto camera_far=render(camera_value,.00390625f,100,far_ratio,label+"-camera-far");
      const auto raw_near=render(raw_value,.015625f,100,near_ratio,label+"-raw-near");
      const auto raw_far=render(raw_value,.00390625f,100,far_ratio,label+"-raw-far");
      equivalent(camera_near,raw_near,"Raw/projection reference paths diverged after a scene change");
      equivalent(camera_far,raw_far,"Raw/projection pairwise strength diverged after a scene change");
      const double expected_span=2*(double(height)/2160*100)*.05*(near_ratio-far_ratio);
      need(std::abs(std::abs(camera_near.disparity-camera_far.disparity)-expected_span)<=.5,
        "Room/wall/return binocular span disagrees with current-zero normalization");
    }
    // Different initial scenes must converge to the same present geometry.
    camera_source different_start(camera::make(0,.0625));
    raw_source raw_different_start(width,height,true);
    different_start.initialize(.015625f);
    raw_different_start.initialize(.015625f);
    for (unsigned sample=0;sample<24;++sample) {
      different_start.capture(.0078125f); raw_different_start.capture(.0078125f);
    }
    equivalent(near_image,render(parameters(different_start.state,different_start.encoding),.015625f,100,2,
      "different-start-same-room-near"),"Initial scene still permanently determines the current stereo strength");
    equivalent(near_image,render(parameters(raw_different_start.state),.015625f,100,2,
      "raw-different-start-same-room-near"),"Initial scene still permanently determines raw stereo strength");
    need(scope.good() && log.good(),"Current-zero evidence was not written completely");
    std::puts("PASS production current-zero normalization: world/raw units, finite-far projection, independent strength/zero geometry and coupled room/wall/return through actual shader");
  }
}

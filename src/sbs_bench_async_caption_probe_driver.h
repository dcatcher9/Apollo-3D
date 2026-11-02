#pragma once
// EXTERNAL UNCOMPILED DRIVER. Requires the proposed estimator hook and a REAL production
// D3D draw/readback callback. Not wired into sbs_bench_harness or the live converter.
// No Boolean "busy" argument, CPU pixel warp, model output, or receipt can be supplied by caller.
#include "video_depth_estimator.h"
#include "host_sbs_held_geometry_current_colour_policy.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace external_actual_async_caption_probe {
  namespace lane = external_held_geometry_current_colour_v2;
  using clock_t = std::chrono::steady_clock;
  struct frame_t {
    ID3D11ShaderResourceView *immutable_full_colour = nullptr;
    std::uint64_t observation_timestamp_us = 0;
    std::uint64_t estimator_frame_id = 0; // Separate from source conversion serial1/2/3.
    std::uint64_t conversion_serial = 0; // Actual submitted source conversion identity, not depth owner.
    lane::stamp_t authored_capture_source_stamp {};
    std::string source_file_sha256;
  };
  enum class phase_e { matched_A, busy_B_current_colour, busy_empty_current_colour, late_B_geometry_current_empty, matched_empty_C };
  struct phase_record_t {
    phase_e phase;
    std::uint64_t current_colour_frame = 0, completed_geometry_frame = 0;
    std::uint64_t current_colour_conversion_serial = 0, geometry_completed_conversion_serial = 0;
    std::uint64_t actual_depth_owner_frame = 0;
    std::optional<std::uint64_t> actual_Base_clock_us;
    lane::stamp_t current_colour_source_stamp {};
    models::host_sbs_gpu_completion_receipt::expected_t geometry_publication;
    std::optional<lane::presentation_receipt_t> presentation_only_receipt;
    std::optional<models::joined_completion_gate_benchmark::snapshot_t> gate_before, gate_after;
    std::optional<models::host_sbs_gpu_completion_receipt::receipt_t> decoded_owner_receipt_after_draw;
    std::string source_file_sha256;
    std::int64_t before_query_return_wall_us=0,after_query_return_wall_us=0;
    // Source stamp belongs to the declared authored input schedule, not a DDup/live capture.
    bool authored_capture_schedule = true, current_cut_unknown_for_held_draw = true;
  };
  struct callbacks_t {
    // MUST record_host_sbs_v2_draw with the current supplied source SRV plus owner Base/state/
    // old joined tuple, then complete a real GPU output readback+hash before returning. For held
    // phases apply typed alive_action to separate scratchM (owner M/SCT never modified). Existing
    // shader authentication/current-RGB match remain. No estimator call inside this callback.
    std::function<void(ID3D11ShaderResourceView *, const models::estimate_result &,
                       const phase_record_t &)> production_draw_and_GPU_readback;
    std::function<void(const phase_record_t &)> append_immutable_trace;
    // Persist all actual policy inputs and selected route BEFORE a rejection can throw.
    std::function<void(const phase_record_t &,const lane::geometry_owner_t &,
                       const lane::captured_colour_t &,const lane::current_evidence_t &)> append_attempt_diagnostic;
  };
  struct options_t {
    bool explicitly_enable_bounded_approximate_research = false;
    std::uint32_t gate_hold_after_entry_ms = 120;
    std::uint32_t empty_capture_after_entry_ms = 33;
    std::uint32_t setup_timeout_ms = 30000;
    std::uint64_t subtitle_context_generation = 0; // Canonical existing offline context.
    models::input_color_space colour_space = models::input_color_space::srgb;
  };
  inline void require(bool yes, const char *reason) { if (!yes) throw std::runtime_error(reason); }
  inline auto checked_gate(models::video_depth_estimator &est) {
    const auto s=est.joined_completion_gate_status_for_benchmark();require(s.has_value(),"Real estimator gate unavailable");return *s;
  }
  struct always_release_t {
    models::video_depth_estimator &est;
    ~always_release_t() { est.release_joined_completion_gate_for_benchmark(); }
  };
  inline models::host_sbs_gpu_completion_receipt::receipt_t prior_receipt(
    models::video_depth_estimator &est, const models::estimate_result &owner
  ) {
    const auto deadline=clock_t::now()+std::chrono::milliseconds {250};
    do {
      const auto p=est.poll_publication_receipts();
      if(p.current_publication && p.current_publication->expected==owner.publication &&
         p.current_publication->depth_cache_authorized()) return *p.current_publication;
      std::this_thread::yield();
    } while(clock_t::now()<deadline);
    throw std::runtime_error("Prior genuine publication receipt unavailable; no fabricated owner allowed");
  }
  inline auto real_busy_query(models::video_depth_estimator &est, models::input_color_space colour) {
    const auto before=checked_gate(est);
    const auto p=est.try_finish_pending_depth_nonblocking(colour,false);
    const auto after=checked_gate(est);
    require(!p.ready,"B became ready before current-source draw; controlled-busy test unavailable");
    require(after.callback_entered && !after.released && !after.callback_timed_out && !after.callback_entry_timed_out,
      "Callback is not presently holding the real root");
    require(after.capture_state_checked && after.capture_state_query_result==CUDA_SUCCESS && after.capture_state==0,
      "One-shot callback enqueue was not outside actual CUDA stream capture");
    require(after.actual_event_queries>before.actual_event_queries &&
      after.actual_not_ready_queries>before.actual_not_ready_queries && after.last_actual_event_query==CUDA_ERROR_NOT_READY,
      "Actual estimator cuEventQuery did not report CUDA_ERROR_NOT_READY");
    return after;
  }
  inline lane::geometry_owner_t make_owner(const models::estimate_result &result,
    const std::optional<models::host_sbs_gpu_completion_receipt::receipt_t> &receipt, const frame_t &colour,
    clock_t::time_point accepted_at, const lane::domain_t &domain) {
    lane::geometry_owner_t owner;
    owner.estimator_generation=result.publication.estimator_generation;
    owner.publication_token=result.publication.publication_sequence;
    owner.completed_colour_conversion_serial=colour.conversion_serial;
    owner.completed_colour_frame=result.completed_frame_id;
    if(receipt) owner.actual_depth_owner_frame=receipt->depth_owner_frame_id;
    owner.completed_colour_source_stamp=colour.authored_capture_source_stamp;owner.accepted_capture_at=accepted_at;
    owner.domain=domain;
    if(receipt) owner.GPU_base_observation_clock_us=receipt->depth_owner_timestamp_us;
    owner.authenticated_completion_and_renderer=models::parallax_v2_result_is_authenticated(result);
    return owner;
  }

  struct prepared_t {
    models::video_depth_estimator *estimator = nullptr;
    models::estimate_result actual_warmup;
    models::host_sbs_gpu_completion_receipt::receipt_t actual_warmup_receipt;
    models::joined_completion_gate_benchmark::snapshot_t actual_gate_after_prepare;
    clock_t::time_point real_prepare_completed_at {};
  };
  inline prepared_t prepare(models::video_depth_estimator &est,
    ID3D11ShaderResourceView *immutable_empty_source,const options_t &options) {
    require(options.explicitly_enable_bounded_approximate_research && immutable_empty_source,
      "Explicit research policy and actual immutable warmup source required");
    models::joined_completion_gate_benchmark::configuration_t config;
    config.enabled=true;config.target_frame_id=3;
    config.hold_after_callback_enter_ms=options.gate_hold_after_entry_ms;
    config.callback_entry_timeout_ms=options.setup_timeout_ms;config.callback_safety_timeout_ms=500;
    require(est.configure_joined_completion_gate_for_benchmark(config),
      "Gate targetB3 configuration must succeed before actual warmup/firstinput");
    try {
      const models::gpu_adaptive_reuse_request force {.observation_timestamp_us=1};
      require(est.estimate_depth(immutable_empty_source,options.colour_space,1,false,{},
        models::depth_optional_work_mode_e::ordinary,force,options.subtitle_context_generation).inference_enqueued,
        "Actual empty warmup did not enqueue");
      auto warmup=est.finish_pending_depth_for_evaluation(options.colour_space,false);
      require(warmup.completed_frame_valid && warmup.completed_frame_id==1 &&
        warmup.publication.observation_timestamp_us==1 && models::parallax_v2_result_is_authenticated(warmup),
        "Actual warmup completion authentication failed");
      const auto receipt=prior_receipt(est,warmup);
      require(receipt.depth_owner_frame_id==1 && receipt.depth_owner_timestamp_us==1,
        "Actual warmup receipt does not name genuine warmup owner1");
      const auto gate=checked_gate(est);
      require(gate.configuration.target_frame_id==3 && !gate.attempted && !gate.queued &&
        !gate.callback_entered && !gate.released && !gate.cancelled && gate.actual_event_queries==0,
        "Gate was consumed/released by preparation");
      return {&est,std::move(warmup),receipt,gate,clock_t::now()};
    } catch(...) {
      est.release_joined_completion_gate_for_benchmark();
      throw;
    }
  }

  inline void run(models::video_depth_estimator &est, const std::array<frame_t,3> &frames,
                  const lane::domain_t &domain, const options_t &options, const callbacks_t &cb,
                  const prepared_t &prepared) {
    require(options.explicitly_enable_bounded_approximate_research,"Private approximate research policy not explicitly enabled");
    require(bool(cb.production_draw_and_GPU_readback)&&bool(cb.append_immutable_trace)&&bool(cb.append_attempt_diagnostic),
      "Real GPU draw/readback, trace and pre-choice diagnostic callbacks required");
    require(options.empty_capture_after_entry_ms>0 && options.empty_capture_after_entry_ms<options.gate_hold_after_entry_ms,
      "Empty authored capture must occur while the gate is independently held");
    for(std::size_t i=0;i<frames.size();++i) {
      require(frames[i].estimator_frame_id==i+2 && frames[i].conversion_serial==i+1 && frames[i].immutable_full_colour && frames[i].observation_timestamp_us>1 && frames[i].authored_capture_source_stamp!=lane::stamp_t {},"Invalid declared input packet");
      if(i) require(frames[i].observation_timestamp_us>frames[i-1].observation_timestamp_us &&
        frames[i].authored_capture_source_stamp>frames[i-1].authored_capture_source_stamp,"Source clocks must be declared, ordered, and distinct");
    }
    require(frames[0].immutable_full_colour!=frames[1].immutable_full_colour &&
      frames[0].immutable_full_colour!=frames[2].immutable_full_colour && frames[1].immutable_full_colour!=frames[2].immutable_full_colour,
      "Keep A/B/empty source textures immutable and separate while B is pending");
    require(prepared.estimator==&est && prepared.actual_warmup.completed_frame_valid &&
      prepared.actual_warmup.completed_frame_id==1 &&
      est.publication_is_current(prepared.actual_warmup.publication) &&
      frames[0].authored_capture_source_stamp>prepared.real_prepare_completed_at,
      "Real preparation must remain current and precede authored epoch");
    const auto warmup_receipt=prior_receipt(est,prepared.actual_warmup);
    require(warmup_receipt.expected==prepared.actual_warmup_receipt.expected &&
      warmup_receipt.depth_owner_frame_id==1 && warmup_receipt.depth_owner_timestamp_us==1,
      "Actual current warmup receipt no longer matches preparation");
    const auto before_A_gate=checked_gate(est);
    require(before_A_gate.configuration.target_frame_id==3 &&
      before_A_gate.configuration.hold_after_callback_enter_ms==options.gate_hold_after_entry_ms &&
      before_A_gate.configuration.callback_safety_timeout_ms==500 &&
      !before_A_gate.attempted && !before_A_gate.queued && !before_A_gate.released &&
      !before_A_gate.cancelled && before_A_gate.actual_event_queries==0,
      "Prepared actual gate was changed/consumed before authoredA");
    always_release_t release_on_every_exit {est};
    const auto submit=[&](std::size_t i) {
      const models::gpu_adaptive_reuse_request force { .observation_timestamp_us=frames[i].observation_timestamp_us };
      return est.estimate_depth(frames[i].immutable_full_colour,options.colour_space,frames[i].estimator_frame_id,false,{},
        models::depth_optional_work_mode_e::ordinary,force,options.subtitle_context_generation);
    };
    std::this_thread::sleep_until(frames[0].authored_capture_source_stamp);
    const auto accepted_A_at=clock_t::now();
    require(submit(0).inference_enqueued,"A force-infer did not enqueue");
    auto A=est.finish_pending_depth_for_evaluation(options.colour_space,false);
    require(A.completed_frame_valid&&A.completed_frame_id==frames[0].estimator_frame_id&&models::parallax_v2_result_is_authenticated(A),"A genuine completion unavailable");
    phase_record_t a;a.phase=phase_e::matched_A;a.current_colour_frame=1;a.completed_geometry_frame=A.completed_frame_id;
    a.current_colour_conversion_serial=frames[0].conversion_serial;a.geometry_completed_conversion_serial=frames[0].conversion_serial;
    a.current_colour_source_stamp=frames[0].authored_capture_source_stamp;a.geometry_publication=A.publication;a.source_file_sha256=frames[0].source_file_sha256;
    cb.production_draw_and_GPU_readback(frames[0].immutable_full_colour,A,a);
    const auto receipt_A=prior_receipt(est,A);
    a.actual_depth_owner_frame=receipt_A.depth_owner_frame_id;a.actual_Base_clock_us=receipt_A.depth_owner_timestamp_us;
    a.decoded_owner_receipt_after_draw=receipt_A;cb.append_immutable_trace(a);
    auto current_owner=make_owner(A,receipt_A,frames[0],accepted_A_at,domain);
    lane::lane_t lane_state;
    // Only the genuine producer generation may initialise this separate policy namespace.
    // Late B completion below must NOT reset the latest current-colour watermark.
    require(lane_state.reset_pipeline(current_owner.estimator_generation),"Cannot initialise genuine estimator generation");
    require(lane_state.adopt_completed(current_owner),"Cannot adopt authentic A owner");
    std::this_thread::sleep_until(frames[1].authored_capture_source_stamp);
    const auto accepted_B_at=clock_t::now();require(submit(1).inference_enqueued,"B force-infer did not enqueue");
    const auto entry_deadline=clock_t::now()+std::chrono::milliseconds {options.setup_timeout_ms};
    auto gate=checked_gate(est);
    while(!gate.callback_entered && clock_t::now()<entry_deadline) {
      require(!gate.callback_entry_timed_out&&!est.has_terminal_failure(),"Actual B root could not reach controlled gate");
      std::this_thread::sleep_for(std::chrono::milliseconds {1});gate=checked_gate(est);
    }
    require(gate.callback_entered&&!gate.released,"Actual callback-entry proof unavailable");
    const auto empty_due=clock_t::time_point {std::chrono::microseconds {gate.entered_wall_us}}+
      std::chrono::milliseconds {options.empty_capture_after_entry_ms};
    const auto held_draw=[&](std::size_t source_index,const models::estimate_result &owner,
      const std::optional<models::host_sbs_gpu_completion_receipt::receipt_t> &owner_receipt,phase_e phase,bool must_be_actual_busy) {
      phase_record_t row;row.phase=phase;row.current_colour_frame=source_index+1;row.completed_geometry_frame=owner.completed_frame_id;
      row.current_colour_conversion_serial=frames[source_index].conversion_serial;row.geometry_completed_conversion_serial=current_owner.completed_colour_conversion_serial;
      if(owner_receipt) {row.actual_depth_owner_frame=owner_receipt->depth_owner_frame_id;row.actual_Base_clock_us=owner_receipt->depth_owner_timestamp_us;}
      row.current_colour_source_stamp=frames[source_index].authored_capture_source_stamp;row.geometry_publication=owner.publication;row.source_file_sha256=frames[source_index].source_file_sha256;
      if(must_be_actual_busy) {
        row.gate_before=real_busy_query(est,options.colour_space);
        row.before_query_return_wall_us=std::chrono::duration_cast<std::chrono::microseconds>(clock_t::now().time_since_epoch()).count();
      }
      require(est.publication_is_current(owner.publication),"Borrowed geometry publication changed before draw");
      lane::captured_colour_t colour {frames[source_index].conversion_serial,frames[source_index].authored_capture_source_stamp,clock_t::now(),domain,true,true};
      lane::current_evidence_t evidence;evidence.joined_root_busy=must_be_actual_busy;
      evidence.publication_still_current=true;evidence.structural_route_join_verified=true;evidence.current_cut=lane::cut_evidence_e::unknown;
      row.presentation_only_receipt=lane_state.choose(colour,evidence,true);
      cb.append_attempt_diagnostic(row,current_owner,colour,evidence);
      require(row.presentation_only_receipt->route==lane::route_e::presentation_only_held_geometry_current_color,"Bounded held-current-colour policy rejected draw");
      cb.production_draw_and_GPU_readback(frames[source_index].immutable_full_colour,owner,row);
      require(est.publication_is_current(owner.publication),"Callback consumed/changed owner during current-colour draw");
      if(must_be_actual_busy) {
        row.gate_after=real_busy_query(est,options.colour_space);
        row.after_query_return_wall_us=std::chrono::duration_cast<std::chrono::microseconds>(clock_t::now().time_since_epoch()).count();
      }
      else if(!owner_receipt) {
        // No fresh clock/receipt readback is required to schedule this draw. The existing GPU
        // renderer validates the real Base/SCT pair. AFTER the actual current-colour GPU readback,
        // independently collect the ordinary completed-publication receipt for diagnostic proof.
        // Do not rewrite the already-selected presentation receipt's unknown optional clock.
        const auto decoded=prior_receipt(est,owner);row.decoded_owner_receipt_after_draw=decoded;
        row.actual_depth_owner_frame=decoded.depth_owner_frame_id;row.actual_Base_clock_us=decoded.depth_owner_timestamp_us;
      }
      cb.append_immutable_trace(row);
    };
    held_draw(1,A,receipt_A,phase_e::busy_B_current_colour,true);
    std::this_thread::sleep_until(std::max(empty_due,frames[2].authored_capture_source_stamp));
    // Critically: NO estimate(C) and NO finish_pending(B) before C was actually drawn/read back.
    held_draw(2,A,receipt_A,phase_e::busy_empty_current_colour,true);
    models::pending_depth_poll_result completed;
    const auto completion_deadline=clock_t::now()+std::chrono::milliseconds {options.setup_timeout_ms};
    do {
      completed=est.try_finish_pending_depth_nonblocking(options.colour_space,false);
      if(completed.ready) break;
      std::this_thread::sleep_for(std::chrono::milliseconds {1});
    } while(clock_t::now()<completion_deadline);
    require(completed.ready&&completed.result.completed_frame_valid&&completed.result.completed_frame_id==frames[1].estimator_frame_id,"Real B completion unavailable after independent release");
    auto B=completed.result;require(models::parallax_v2_result_is_authenticated(B),"B completion failed authentication");
    // A's borrowed aliases may now contain B. Never draw A again after this normalization.
    // Keep GPU Base clock diagnostic unknown until after this actual C-colour draw. Never
    // fabricate a B receipt or stall fresh colour to synchronously read a clock before drawing.
    current_owner=make_owner(B,std::nullopt,frames[1],accepted_B_at,domain);
    require(lane_state.adopt_completed(current_owner),"Cannot adopt authentic B owner");
    held_draw(2,B,std::nullopt,phase_e::late_B_geometry_current_empty,false);
    require(submit(2).inference_enqueued,"C force-infer did not enqueue after B retired");
    auto C=est.finish_pending_depth_for_evaluation(options.colour_space,false);
    require(C.completed_frame_valid&&C.completed_frame_id==frames[2].estimator_frame_id&&models::parallax_v2_result_is_authenticated(C),"Current empty C completion failed");
    phase_record_t c;c.phase=phase_e::matched_empty_C;c.current_colour_frame=3;c.completed_geometry_frame=C.completed_frame_id;
    c.current_colour_conversion_serial=frames[2].conversion_serial;c.geometry_completed_conversion_serial=frames[2].conversion_serial;c.geometry_publication=C.publication;c.current_colour_source_stamp=frames[2].authored_capture_source_stamp;c.source_file_sha256=frames[2].source_file_sha256;
    cb.production_draw_and_GPU_readback(frames[2].immutable_full_colour,C,c);
    const auto receipt_C=prior_receipt(est,C);c.decoded_owner_receipt_after_draw=receipt_C;
    c.actual_depth_owner_frame=receipt_C.depth_owner_frame_id;c.actual_Base_clock_us=receipt_C.depth_owner_timestamp_us;cb.append_immutable_trace(c);
    const auto final=checked_gate(est);
    require(final.callback_exited&&!final.callback_timed_out&&!final.callback_entry_timed_out&&final.actual_not_ready_queries>=4&&final.actual_ready_queries>=1,
      "Actual root gate/query/release proof incomplete");
  }
}

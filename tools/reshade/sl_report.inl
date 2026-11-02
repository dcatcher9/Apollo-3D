// SPDX-License-Identifier: GPL-3.0-only
// Part of streamline_camera_probe.cpp (sl_adapter): existing shipping report
// functions and formatting fragments. The shell retains namespace state; the
// tag/state rate statics retain their original owning functions and guards.
// Helpers receive existing values/references and execute at the old log points.
// One translation unit, included at shell end. Camera reporting keeps its anonymous
// namespace rather than the runtime-test private view.

namespace sunshine_streamline {
  SUNSHINE_PROBE_INTERNAL {
    // A declared Resource size that disagrees with the allocation is ignored
    // when the tag's extent defines the region (depth_capture::source_region);
    // name it once per process.
    void report_declared_size(const evaluated_depth_tag &tag) {
      static std::atomic<bool> reported{};
      const auto &value = tag.value;
      std::uint32_t width{}, height{};
      if (sunshine_api_contract::null_extent(value.area) || reported.load(std::memory_order_relaxed) ||
          !depth_capture::source_extent(tag.direct_source, width, height) ||
          !((value.resource_width && value.resource_width != width) || (value.resource_height && value.resource_height != height)) ||
          reported.exchange(true, std::memory_order_relaxed)) return;
      char text[320]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline depth: declared resource size %ux%u differs from the %ux%u allocation; the tag extent %u,%u %ux%u "
        "defines the region and the declared size is ignored (tag_type=%u viewport=%u; logged once)",
        value.resource_width, value.resource_height, width, height, value.area.left, value.area.top, value.area.width,
        value.area.height, value.type, value.viewport);
      message(text);
    }
    const char *name(frame_identity_kind value) {
      switch (value) {
        case frame_identity_kind::unavailable: return "unavailable";
        case frame_identity_kind::v1_numeric: return "v1_numeric";
        case frame_identity_kind::v2_observed_token: return "v2_observed_token";
        case frame_identity_kind::v2_constants_call: return "v2_constants_call";
      }
      return "unknown";
    }
    // Why an FG tag boundary named no capturable depth, at most every 5 s.
    // The tag described is the call's first present depth tag in priority
    // order; reason is its first failing gate. A live FG scope is withdrawn,
    // unless DLSS-G's own evaluation supplies its copies (dlssg_supplied); a
    // silent one has no FG capture to withdraw. Either way the frame has no
    // FG copy, so the scope owns no depth for it (miss_frame_generation).
    // Diagnostic only.
    void report_fg_refusal(const evaluation_snapshot &snapshot, const char *reason, bool withdrawn, bool live,
        bool supplied) {
      const auto count = fg_refusals.fetch_add(1, std::memory_order_relaxed) + 1;
      const auto now = sunshine_game3d::clock::game_now();
      auto next = next_fg_refusal_report.load(std::memory_order_relaxed);
      if (now < next || !next_fg_refusal_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) return;
      static const evaluated_depth_tag none;
      const auto *tag = &none;
      for (const unsigned index : depth_priority) if (snapshot.tags[index].present) { tag = &snapshot.tags[index]; break; }
      char text[720]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline FG depth refused: reason=%s viewport=%u tag_type=%u scope=%s lifecycle=%u native=0x%llx "
        "command=0x%llx frame=%s status=%s fg=%u fg_scope=%s withdrawn=%u dlssg_supplied=%u refusals=%llu; this FG tag call "
        "names no capturable depth; %s",
        reason, snapshot.viewport, tag->present ? tag->value.type : UINT32_MAX,
        !tag->present ? "absent" : tag->scope == tag_scope::active_global ? "global" :
          tag->scope == tag_scope::explicit_frame ? "frame" : "local",
        tag->present ? tag->value.lifecycle : UINT32_MAX, static_cast<unsigned long long>(tag->value.native_resource),
        static_cast<unsigned long long>(snapshot.command_buffer), name(snapshot.frame.kind), name(snapshot.status),
        unsigned(snapshot.frame_generation_input), live ? "live" : "silent", unsigned(withdrawn), unsigned(supplied),
        static_cast<unsigned long long>(count),
        withdrawn ? "the viewport's FG depth source is withdrawn, and the FG scope owns no depth for this frame, so the "
          "viewport's ordinary SR/RR depth may supply it" :
          supplied ? "DLSS-G's own NGX evaluation copies this FG scope's depth (DLSSG.Depth), so nothing is withdrawn" :
          "the FG scope owns no depth for this frame, so nothing is withdrawn and the viewport's ordinary SR/RR depth "
          "may supply it");
      message(text);
    }
    // A depth copy recorded inside a frame-generation tag call (diagnostic:
    // the recorded candidate's): counted, and with it a copy made from the
    // state the call's list recorded over a declaration of the tag's later
    // use that named another (S11). The bounded line comes at most every 5 s,
    // only after such a copy.
    void count_tag_call_copy(const depth_capture::input *candidates, std::size_t count,
        const depth_capture::record_diagnostic &diagnostic) {
      tag_call_copies.copied.fetch_add(1, std::memory_order_relaxed);
      bool over = false;
      for (std::size_t i = 0; i != count && !over; ++i)
        over = candidates[i].resource.native == diagnostic.resource &&
          candidates[i].proof == sunshine_scene_depth::state_proof::declared_for_use && diagnostic.used_observed_state;
      if (!over) return;
      const auto over_total = tag_call_copies.recorded_over_declared.fetch_add(1, std::memory_order_relaxed) + 1;
      const auto now = sunshine_game3d::clock::game_now();
      auto next = next_tag_call_report.load(std::memory_order_relaxed);
      if (now < next || !next_tag_call_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) return;
      char text[480]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline FG tag-call copies: copied=%llu recorded_over_declared=%llu state=0x%x declared=0x%x; a "
        "ValidUntilPresent or ValidUntilEvaluate declaration describes the tag's later use, so a tag call copies its "
        "tag from the state the call's list records",
        static_cast<unsigned long long>(tag_call_copies.copied.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(over_total), diagnostic.copy_state, diagnostic.native_state);
      message(text);
    }
    // The bounded deferral line, at most every 5 s and only after its totals
    // changed: what the copies deferred to their frame's Present (R2) became.
    void report_deferrals() {
      const auto activity = deferrals.depth.load(std::memory_order_relaxed) + deferrals.ui.load(std::memory_order_relaxed) +
        deferrals.cancelled.load(std::memory_order_relaxed) + deferrals.lapsed.load(std::memory_order_relaxed);
      const auto now = sunshine_game3d::clock::game_now();
      auto next = next_deferral_report.load(std::memory_order_relaxed);
      if (activity == reported_deferrals.load(std::memory_order_relaxed) || now < next ||
          !next_deferral_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) return;
      reported_deferrals.store(activity, std::memory_order_relaxed);
      char text[640]{};
      std::snprintf(text, sizeof(text),
        "Sunshine Streamline deferred tag copies: depth={deferred=%llu copied=%llu refused=%llu last_refusal=%s} "
        "ui={deferred=%llu copied=%llu refused=%llu} cancelled_by_untag=%llu lapsed=%llu waited_for_frame=%llu "
        "pending=%u; a ValidUntilPresent tag whose call has no command list, or a frame-generation or UI input whose "
        "call's list records no state of it or a write state, is copied on the Present queue right before its frame's "
        "Present",
        static_cast<unsigned long long>(deferrals.depth.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.depth_copied.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.depth_refused.load(std::memory_order_relaxed)),
        depth_capture::name(static_cast<depth_capture::record_stage>(deferrals.refusal.load(std::memory_order_relaxed))),
        static_cast<unsigned long long>(deferrals.ui.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.ui_copied.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.ui_refused.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.cancelled.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.lapsed.load(std::memory_order_relaxed)),
        static_cast<unsigned long long>(deferrals.waited.load(std::memory_order_relaxed)),
        deferral_pending.load(std::memory_order_relaxed));
      message(text);
    }
    void report_tag_resource(const batch &out, const abi_v2::resource_tag &tag, origin source,
        const tag_record &value, const abi_v2::resource &resource, unsigned index, bool zero_resource_header) {
      // A null Resource also reads as the zero header; only a real native is
      // the compatibility layout. An untag is reported on its own line.
      if (value.supported && zero_resource_header && index < 3 && tag.resource_ptr && value.value.native_resource) {
        const auto now = sunshine_game3d::clock::game_now();
        auto next = next_compatibility_report.load(std::memory_order_relaxed);
        if (now >= next && next_compatibility_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) {
          char text[384]{};
          std::snprintf(text, sizeof(text),
            "Sunshine Streamline zero-base resource compatibility: tag_type=%u viewport=%u native=0x%llx type=%u state=0x%x state_proof=%s declared_extent=%ux%u native_retained=%u",
            tag.type, out.viewport, static_cast<unsigned long long>(value.value.native_resource), resource.type, resource.state,
            value.state_requires_observation ? "requires-observed-state" : "declared", resource.width, resource.height,
            value.direct_source ? 1u : 0u);
          message(text);
        }
      }
      if (value.supported && index < 3 && !value.value.native_resource) {
        const auto count = depth_untags.fetch_add(1, std::memory_order_relaxed) + 1;
        const auto now = sunshine_game3d::clock::game_now();
        auto next = next_untag_report.load(std::memory_order_relaxed);
        if (now >= next && next_untag_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) {
          char text[320]{};
          std::snprintf(text, sizeof(text),
            "Sunshine Streamline depth untag: tag_type=%u viewport=%u scope=%s lifecycle=%u resource=%s untags=%llu; "
            "a depth tag without a native resource ends that tag and names no depth source",
            tag.type, out.viewport, source == origin::global ? "global" : source == origin::frame ? "frame" : "local",
            tag.lifecycle, tag.resource_ptr ? "null_native" : "null", static_cast<unsigned long long>(count));
          message(text);
        }
      }
    }
    void report_rejected_tag(const batch &out, const abi_v2::resource_tag &tag, const tag_record &value,
        const abi_v2::resource &resource, bool precision_ok, bool resource_ok, std::atomic<std::uint64_t> &next_tag_report) {
        const auto now = sunshine_game3d::clock::game_now();
        auto next = next_tag_report.load(std::memory_order_relaxed);
        if (now >= next && next_tag_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) {
          char text[700]{};
          std::snprintf(text, sizeof(text),
            "Sunshine Streamline rejected depth tag: refusal=%s tag_type=%u viewport=%u tag_guid=%08x tag_version=%llu tag_extension=0x%llx precision_supported=%u lifecycle=%u resource_read=%u resource_guid=%08x resource_version=%llu resource_type=%u resource_extension=0x%llx declared_native=0x%llx state=0x%x declared_extent=%ux%u",
            name(value.value.refusal), tag.type, out.viewport, tag.base.type.a, static_cast<unsigned long long>(tag.base.version),
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(tag.base.next)), precision_ok, tag.lifecycle, resource_ok,
            resource.base.type.a, static_cast<unsigned long long>(resource.base.version), resource.type,
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(resource.base.next)),
            static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(resource.native)), resource.state, resource.width, resource.height);
          message(text);
        }
    }
    void report_v1_state_lookup(const evaluation_snapshot &snapshot, const evaluated_depth_tag &tag,
        const depth_capture::input &value, const v1_resource_state::read_details &state_details,
        std::atomic<std::uint64_t> &next_state_report) {
          const auto now = sunshine_game3d::clock::game_now();
          auto next = next_state_report.load(std::memory_order_relaxed);
          if (now >= next && next_state_report.compare_exchange_strong(next, now + 5000, std::memory_order_relaxed)) {
            char text[640]{};
            std::snprintf(text, sizeof(text),
              "Sunshine Streamline V1 state lookup: %s; resource=0x%llx tag=%u viewport=%u sequence=%llu; %llux%u format=%u mips=%u layers=%u samples=%u; hr=0x%08lx bytes=%u chi=0x%x; tagged=0x%x; requires observed state",
              state_details.stage, static_cast<unsigned long long>(tag.value.native_resource), tag.value.type, snapshot.viewport,
              static_cast<unsigned long long>(snapshot.sequence), static_cast<unsigned long long>(state_details.description.Width),
              state_details.description.Height, unsigned(state_details.description.Format), state_details.description.MipLevels,
              state_details.description.DepthOrArraySize, state_details.description.SampleDesc.Count,
              static_cast<unsigned long>(state_details.result), state_details.bytes, state_details.encoded, value.native_state);
            message(text);
          }
    }
  }
  namespace {
    void report_camera_availability(std::uint64_t now) {
      if (!sunshine_upscaler_trace::capture_enabled() || !hooks_installed) return;
      static std::uint64_t camera_report_at{}, camera_report_count{UINT64_MAX};
      auto camera_pin = now >= camera_report_at ? read_metadata() : metadata_owner::read_pin{};
      if (camera_pin) {
        unsigned cameras = 0, valid_recent = 0;
        camera_data camera_sample;
        camera_key camera_sample_key;
        camera_validation camera_sample_validation;
        std::uint64_t camera_sample_tick{};
        struct depth_tag_sample {
          depth_tag value;
          std::uint64_t tick{};
          origin scope{};
          bool present{}, supported{};
        };
        std::array<depth_tag_sample, 3> depth_samples{};
        for (const auto &record : camera_pin->records) if (record.used && record.has_camera) {
          ++cameras;
          const auto checked = validate(record.camera);
          valid_recent += sunshine_freshness::fresh_le(record.camera_tick, now, sunshine_freshness::camera_age_ms) &&
            record.decoded == decode_status::ok && checked.valid();
          if (record.decoded == decode_status::ok && record.camera_tick >= camera_sample_tick) {
            camera_sample = record.camera;
            camera_sample_key = record.key;
            camera_sample_validation = checked;
            camera_sample_tick = record.camera_tick;
            depth_samples = {};
            for (const auto *group : {&record.active_tags, &record.global_tags, &record.local_tags}) {
              for (const auto &tag : *group) {
                if (!tag.present || !buffers::is_depth(tag.value.type)) continue;
                auto &sample = depth_samples[slot_of(tag.value.type)];
                if (!sample.present || tag.tick > sample.tick)
                  sample = {tag.value, tag.tick, tag.source, tag.present, tag.supported};
              }
            }
          }
        }
        const auto calls = constant_calls.load(std::memory_order_relaxed);
        camera_pin.reset();
        camera_report_at = now + 5000;
        if (calls != camera_report_count) {
          // vram_warnings: SDK calls answered with eWarnOutOfVRAM since the
          // observation began, each accepted as a success.
          char availability[352]{};
          std::snprintf(availability, sizeof(availability),
            "Sunshine Streamline camera availability: constants_calls=%llu cameras=%u recent_valid_projection=%u vram_warnings=%llu; NGX_frame_match=unproven; availability does not authorize cross-API scale",
            static_cast<unsigned long long>(calls), cameras, valid_recent,
            static_cast<unsigned long long>(vram_warnings.load(std::memory_order_relaxed)));
          message(availability);
          if (camera_sample_tick) {
            char detail[1800]{};
            std::snprintf(detail, sizeof(detail),
              "Sunshine Streamline camera sample: viewport=%u frame=%llu explicit_frame=%u age_ms=%llu projection=%s A=%.9g B=%.9g matrix_near=%.9g matrix_far=%.9g infinite_far=%u checked_fov_rad=%.9g checked_aspect=%.9g declared_near=%.9g declared_far=%.9g declared_fov_rad=%.9g declared_aspect=%.9g inverted=%u orthographic=%u reset=%u; checked fields may be incomplete on failure; observation only, NGX_frame_match=unproven",
              camera_sample_key.viewport, static_cast<unsigned long long>(camera_sample_key.frame_key),
              camera_sample_key.explicit_frame ? 1u : 0u,
              static_cast<unsigned long long>(camera_sample_tick <= now ? now - camera_sample_tick : 0),
              name(camera_sample_validation.status), camera_sample_validation.depth_offset,
              camera_sample_validation.depth_scale, camera_sample_validation.near_from_matrix,
              camera_sample_validation.far_from_matrix, camera_sample_validation.infinite_far ? 1u : 0u,
              camera_sample_validation.fov_from_matrix, camera_sample_validation.aspect_from_matrix,
              camera_sample.near_plane, camera_sample.far_plane, camera_sample.fov, camera_sample.aspect,
              unsigned(camera_sample.depth_inverted), unsigned(camera_sample.orthographic),
              unsigned(camera_sample.reset));
            message(detail);
            char tag_details[3][480]{};
            for (unsigned i = 0; i < depth_samples.size(); ++i) {
              const auto &sample = depth_samples[i];
              const auto &tag = sample.value;
              std::snprintf(tag_details[i], sizeof(tag_details[i]),
                "type=%u present=%u supported=%u scope=%u native=0x%llx declared_size=%ux%u area_xywh=%u,%u,%u,%u lifetime=%u age_ms=%llu tag_viewport=%u frame=%llu explicit_frame=%u",
                observed_tag_types[i], sample.present ? 1u : 0u,
                sample.supported ? 1u : 0u, static_cast<unsigned>(sample.scope),
                static_cast<unsigned long long>(tag.native_resource), tag.resource_width, tag.resource_height,
                tag.area.left, tag.area.top, tag.area.width, tag.area.height, tag.lifecycle,
                static_cast<unsigned long long>(sample.present && sample.tick <= now ? now - sample.tick : UINT64_MAX),
                tag.viewport, static_cast<unsigned long long>(tag.frame_key), tag.explicit_frame ? 1u : 0u);
            }
            std::snprintf(detail, sizeof(detail),
              "Sunshine Streamline depth tag sample: camera_viewport=%u depth={%s} highres_depth={%s} linear_depth={%s}; observation only, NGX_resource_overlap=unproven",
              camera_sample_key.viewport, tag_details[0], tag_details[1], tag_details[2]);
            message(detail);
          }
          camera_report_count = calls;
        }
      }
    }
  }
}

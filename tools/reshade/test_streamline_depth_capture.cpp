// SPDX-License-Identifier: GPL-3.0-only
// White-box regressions of streamline_depth_capture.cpp's capture owner, run
// by the probe and diagnostic-texture tests (streamline_depth_capture.h
// testing::). They reach its internals through its test-access header.
#ifndef SUNSHINE_STREAMLINE_PROBE_TEST
#error "Built only into the Streamline probe tests' view of the production sources"
#endif
#include "streamline_depth_capture_test_access.h"
#include "test_streamline_depth_capture_fixture.h"
#include "streamline_depth_capture.h"
#include "capture_state_policy.h"
#include "streamline_native_observer.h"
#include "native_command_storage.h"
#include "native_resource_identity.h"
#include "depth_view_format.h"
#include "depth_cache_update.h" // Selection regressions apply the provider's display policy.
#include "../../src/game3d_debug_formats.h"
#include <d3d12.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <optional>
#include <cstdio>
#include <initializer_list>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace sunshine_streamline::depth_capture {
  namespace testing {
    void age_idle_storage(std::uint64_t ms) {
      std::lock_guard lock(mutex);
      for (auto &value : diagnostic_slots)
        if (!value.id && value.texture) value.idle_tick -= std::min(value.idle_tick, ms);
    }
    bool diagnostic_snapshot_regression() {
      // Same admission helper as production; completion alone must not publish
      // a texture while its recorded copy can still legally replay.
      slot value;
      value.id = 1; value.diagnostic_only = value.preservation_only = true;
      value.command = 3; value.producer_recording = std::make_shared<sunshine_native_command::recording_lifetime>();
      value.producer_recording->cookie = 3;
      queue_progress complete{11, 9, true};
      if (diagnostic_admission(value, complete) != status::recorded) return false;
      value.finished = value.success = true;
      producer_submitted(value, 5, 9, true);
      if (diagnostic_admission(value, complete) != status::submitted) return false;
      retire_recording(value, 3);
      if (diagnostic_admission(value, {11, 8, true}) != status::submitted ||
          diagnostic_admission(value, complete) != status::ready) return false;
      if (diagnostic_admission(value, {11, UINT64_MAX, true}) != status::failed ||
          diagnostic_admission(value, {}) != status::failed) return false;
      value.diagnostic_released = true;
      if (diagnostic_admission(value, complete) != status::stale) return false;
      value.diagnostic_released = false;
      producer_submitted(value, 5, 10, false);
      if (diagnostic_admission(value, complete) != status::failed || reclaimable(value)) return false;
      // Rejection/cancellation cannot turn unknown retirement into a free slot.
      value.diagnostic_released = true;
      if (reclaimable(value)) return false;
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D; desc.Width = 64; desc.Height = 48;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      // Typeless allocations resolve as auxiliary_snapshot_format does: the
      // same-bits UNORM families without a declaration, every other one only
      // through a declared view of its family.
      for (const auto format : {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R16G16B16A16_FLOAT,
          DXGI_FORMAT_R8G8B8A8_TYPELESS, DXGI_FORMAT_B8G8R8A8_TYPELESS, DXGI_FORMAT_B8G8R8X8_TYPELESS,
          DXGI_FORMAT_R10G10B10A2_TYPELESS, DXGI_FORMAT_A8_UNORM}) {
        desc.Format = format;
        if (!diagnostic_description(desc, 0)) return false;
      }
      for (const auto format : {DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R16G16B16A16_TYPELESS,
          DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_NV12, DXGI_FORMAT_BC1_UNORM}) {
        desc.Format = format;
        if (diagnostic_description(desc, 0)) return false;
      }
      for (const auto [format, declared] : {std::pair{DXGI_FORMAT_R16G16B16A16_TYPELESS, DXGI_FORMAT_R16G16B16A16_UNORM},
          std::pair{DXGI_FORMAT_R8_TYPELESS, DXGI_FORMAT_R8_UNORM}, std::pair{DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_FLOAT}}) {
        desc.Format = format;
        if (!diagnostic_description(desc, declared) || diagnostic_description(desc, DXGI_FORMAT_R8G8B8A8_UNORM)) return false;
      }
      desc.Format = DXGI_FORMAT_R8_UNORM; desc.SampleDesc.Count = 2;
      if (diagnostic_description(desc, 0)) return false;
      // Cached allocations count toward the live budget with live tickets, and
      // only the cache can be evicted to admit a changed output layout.
      constexpr auto limit = diagnostic_byte_limit;
      std::array<slot, diagnostic_slot_limit> cache;
      for (unsigned i = 0; i != 5; ++i) {
        cache[i].texture = std::make_shared<texture_reference>();
        cache[i].texture->allocation_bytes = limit / 4;
      }
      cache[0].id = 1; cache[0].diagnostic_reusable = true; // A live ticket survives budget pressure.
      // A small request keeps the fixed limit: 5/4 allocated, room for 1/16.
      if (!reserve_diagnostic_bytes(cache, &cache[9], limit / 16, true, 0, nullptr) ||
          !cache[0].texture || cache[1].texture || cache[2].texture || !cache[3].texture || !cache[4].texture) return false;
      // A large live request is sized from itself: room for eight of it.
      if (!reserve_diagnostic_bytes(cache, &cache[9], limit, true, 0, nullptr) || !cache[3].texture || !cache[4].texture) return false;
      // Dump copies have their own fixed budget: live storage never counts
      // against them and is never evicted for them.
      cache[5].texture = std::make_shared<texture_reference>();
      cache[5].texture->allocation_bytes = limit / 2;
      cache[5].id = 2; // A dump ticket.
      if (!reserve_diagnostic_bytes(cache, &cache[9], limit / 2, false, 0, nullptr) ||
          reserve_diagnostic_bytes(cache, &cache[9], limit / 2 + 1, false, 0, nullptr) ||
          !cache[3].texture || !cache[4].texture) return false;
      return !reserve_diagnostic_bytes(cache, &cache[0], 0, false, 0, nullptr) &&
        !reserve_diagnostic_bytes(cache, &cache[5], limit + 1, false, 0, nullptr) &&
        reserve_diagnostic_bytes(cache, &cache[5], limit, false, 0, nullptr);
    }
    bool record_diagnostic_regression() {
      // COM Release can synchronously invoke observers. Detach ownership first
      // so reentrant reset/put cannot release the same reference twice.
      struct reentrant_release {
        com_ptr<reentrant_release> *owner{};
        unsigned calls{};
        bool saw_empty{};
        void Release() {
          ++calls;
          saw_empty = !owner->p;
          if (calls == 1) owner->put();
        }
      };
      static_assert(!std::is_copy_constructible_v<com_ptr<reentrant_release>>);
      static_assert(!std::is_copy_assignable_v<com_ptr<reentrant_release>>);
      reentrant_release object;
      {
        com_ptr<reentrant_release> owner;
        object.owner = &owner;
        owner.p = &object;
        if (*owner.put() || object.calls != 1 || !object.saw_empty) return false;
        object.calls = 0;
        owner.p = &object;
      }
      if (object.calls != 1 || !object.saw_empty) return false;
      // Exercise real early rejection paths; no fake graphics vtable or GPU is
      // needed. A later caller can overwrite the shared status, not this result.
      initialize(true);
      input value;
      // An FG presentation can advance the device's Present generation while
      // the render thread is inside slSetTag. That cannot expire OnlyValidNow's
      // synchronous copy, while an ordinary retained tag still expires.
      value.source_present_generation = 7;
      for (const auto lifetime : {sunshine_scene_depth::lifetime::until_present,
          sunshine_scene_depth::lifetime::until_evaluation}) {
        value.valid_until = lifetime;
        if (!source_lifetime_current(value, 7) || source_lifetime_current(value, 8)) return false;
      }
      value.valid_until = sunshine_scene_depth::lifetime::at_call;
      value.force_snapshot = true;
      if (!source_lifetime_current(value, 8)) return false;
      value.force_snapshot = false;
      if (source_lifetime_current(value, 8)) return false;
      // The same holds for an UntilPresent tag copied inside its own tag call
      // (FG input), but not for that tag copied later at an evaluation.
      value.valid_until = sunshine_scene_depth::lifetime::until_present;
      value.at_tag_call = true;
      if (!source_lifetime_current(value, 8)) return false;
      value.at_tag_call = false;
      if (source_lifetime_current(value, 8)) return false;
      value = {};
      value.epoch = value.sequence = 1;
      value.resource.native = 7;
      record_diagnostic missing;
      missing.recording_closed = true; missing.width = 123;
      if (record(5, value, &missing) || missing.result != status::malformed ||
          missing.stage != record_stage::missing_source || missing.command != 5 || missing.resource != 7 ||
          missing.recording_closed || missing.width) return false;
      value.source = std::make_shared<source_reference>();
      record_diagnostic lifetime;
      if (record(5, value, &lifetime) || lifetime.result != status::unsupported_lifetime ||
          lifetime.stage != record_stage::unsupported_lifetime || last_status() != status::unsupported_lifetime ||
          missing.result != status::malformed || missing.stage != record_stage::missing_source) return false;
      // The optional observer does not change the rejection or shared status.
      if (record(5, value) || last_status() != lifetime.result) return false;
      value.valid_until = sunshine_scene_depth::lifetime::until_evaluation;
      // An undeclared state needs an observation (capture_state_policy.h);
      // only an unknown proof value has no rule.
      value.proof = static_cast<sunshine_scene_depth::state_proof>(99);
      record_diagnostic proof;
      if (record(5, value, &proof) || proof.result != status::unsupported_state ||
          proof.stage != record_stage::unsupported_proof) return false;
      shutdown();
      record_diagnostic inactive;
      if (record(5, value, &inactive) || inactive.result != status::malformed || inactive.stage != record_stage::inactive) return false;
      if (retain_source(7, &inactive) || inactive.result != status::inactive || inactive.stage != record_stage::inactive ||
          inactive.resource != 7 || inactive.command) return false;
      return true;
    }
    bool crop_region_regression() {
      D3D12_RESOURCE_DESC desc{};
      desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      desc.Width = 2228; desc.Height = 1256;
      desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
      desc.Format = DXGI_FORMAT_R32_FLOAT;
      sunshine_scene_depth::resource_description resource;
      copy_region region;
      if (capture_region(desc, resource, region) != status::ready || region.area.left || region.area.top ||
          region.area.width != 2228 || region.area.height != 1256 || region.width != 2228 || region.height != 1256) return false;
      resource.area = {0, 0, 2227, 1252};
      if (capture_region(desc, resource, region) != status::ready || region.width != 2228 || region.height != 1256 ||
          region.area.left || region.area.top || region.area.width != 2227 || region.area.height != 1252) return false;
      resource.area = {1, 2, 2227, 1252};
      if (capture_region(desc, resource, region) != status::ready || region.area.left != 1 ||
          region.area.top != 2 || region.area.width != 2227 || region.area.height != 1252 ||
          region.width != 2228 || region.height != 1256) return false;
      resource.area.left = UINT32_MAX;
      if (capture_region(desc, resource, region) != status::malformed) return false;
      resource.area = {0, 0, 2228, 0};
      if (capture_region(desc, resource, region) != status::malformed) return false;
      resource.area = {1, 0, 0, 0};
      if (capture_region(desc, resource, region) != status::malformed) return false;
      resource.area = {}; resource.width = 2227;
      if (capture_region(desc, resource, region) != status::malformed) return false;
      // An extent inside the allocation alone defines the region: a declared
      // size that disagrees with the allocation (a render size) is ignored.
      resource.area = {0, 0, 1920, 1080};
      for (const auto declared : {std::array<std::uint32_t, 2>{1920, 1080}, std::array<std::uint32_t, 2>{2228, 1256},
          std::array<std::uint32_t, 2>{2227, 0}}) {
        resource.width = declared[0]; resource.height = declared[1];
        if (capture_region(desc, resource, region) != status::ready || region.width != 2228 || region.height != 1256 ||
            region.area.width != 1920 || region.area.height != 1080) return false;
      }
      resource.area = {}; resource.width = 0; resource.height = 1080;
      if (capture_region(desc, resource, region) != status::malformed) return false;
      resource.height = 0;
      if (capture_region(desc, resource, region) != status::ready || region.area.width != 2228) return false;
      resource.width = 0;
      const auto ordinary = desc;
      for (const auto format : {DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
          DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_D16_UNORM}) {
        desc = ordinary; desc.Format = format;
        resource.area = {};
        if (capture_region(desc, resource, region) != status::ready ||
            region.width != 2228 || region.height != 1256 || region.area.width != 2228 || region.area.height != 1256) return false;
        resource.area = {1, 2, 2227, 1252};
        if (capture_region(desc, resource, region) != status::ready ||
            region.width != 2228 || region.height != 1256 || region.area.left != 1 || region.area.top != 2 ||
            region.area.width != 2227 || region.area.height != 1252) return false;
      }
      desc = ordinary; desc.Format = DXGI_FORMAT_R32_TYPELESS;
      if (capture_region(desc, resource, region) != status::ready) return false;
      desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
      if (capture_region(desc, resource, region) != status::ready || region.width != 2228 || region.height != 1256 ||
          region.area.left != 1 || region.area.top != 2 || region.area.width != 2227 || region.area.height != 1252) return false;
      resource.area = {};
      if (capture_region(desc, resource, region) != status::ready) return false;
      // A mip chain copies mip 0 into a single-mip snapshot; R16_FLOAT is depth.
      desc = ordinary; desc.MipLevels = 6;
      if (capture_region(desc, resource, region) != status::ready || region.width != 2228 || region.height != 1256) return false;
      desc = ordinary; desc.Format = DXGI_FORMAT_R16_FLOAT;
      if (capture_region(desc, resource, region) != status::ready) return false;
      desc = ordinary; desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
      if (capture_region(desc, resource, region) != status::unsupported_resource) return false;
      desc = ordinary; desc.DepthOrArraySize = 2;
      if (capture_region(desc, resource, region) != status::unsupported_resource) return false;
      desc = ordinary; desc.SampleDesc.Count = 2;
      if (capture_region(desc, resource, region) != status::unsupported_resource) return false;
      sunshine_scene_depth::projection_basis projection;
      projection.reversed = true;
      projection.depth_offset = projection.depth_scale = NAN;
      if (!valid_projection(projection)) return false; // Opaque absent coefficients are never consumed.
      projection.supplied = true;
      if (valid_projection(projection)) return false;
      projection = {0, .0625, true, true};
      if (!valid_projection(projection)) return false;
      projection.reversed = false;
      if (valid_projection(projection)) return false;
      projection.depth_scale = -.0625;
      if (!valid_projection(projection)) return false;
      projection = {};
      projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
      projection.direction_supplied = true;
      if (!valid_projection(projection)) return false; // Linear distance needs no camera matrix.
      projection.reversed = true;
      if (valid_projection(projection)) return false;
      projection.raw_scale = -2;
      if (!valid_projection(projection)) return false;
      projection.raw_scale = 0;
      if (valid_projection(projection)) return false;
      projection.raw_scale = -2; projection.raw_bias = NAN;
      if (valid_projection(projection)) return false;
      projection.raw_bias = 0; projection.direction_supplied = false;
      return !valid_projection(projection);
    }
    bool provider_admission_regression() {
      // Exercise the production acquisition policy without COM/GPU operations.
      // Pixel ownership/fences remain covered by the native runtime fixtures.
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        std::array<evaluation_namespace, provider_count> saved_evaluations{evaluations};
        std::array<status, provider_count> saved_attempts{attempt_status[0].load(), attempt_status[1].load()};
        bool saved_requested{requested.load()};
        status saved_reason{reason.load()};
        std::uint64_t saved_serial{serial.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = saved_attempts[i];
          requested = saved_requested; reason = saved_reason; serial = saved_serial;
        }
      } restore;
      fixture::clear_captures(); requested = true;
      {
        // Handoff evidence (evaluation_live): any real evaluation, Streamline
        // SR's source 0 included (FG Off in a Streamline-only game holds the
        // queue in mono while SR's first capture is pending); never the
        // malformed-input marker, which alone uses epoch 0.
        begin_evaluation(5, 1, 0, provider_kind::streamline);
        if (!evaluation_live(GetTickCount64(), 250)) return false;
        evaluations = {};
        begin_evaluation(0, 2, UINT32_MAX, provider_kind::streamline);
        if (evaluation_live(GetTickCount64(), 250)) return false;
        // An established Streamline SR owner (source 0) stays associated
        // through its own epoch's evaluations only, never through the marker.
        queue_state sr;
        sr.provider_established = true; sr.provider = provider_kind::streamline; sr.nominated_epoch = 5;
        expire_association(sr, GetTickCount64());
        if (sr.provider_established) return false;
        evaluations = {};
        begin_evaluation(5, 3, 0, provider_kind::streamline);
        sr.provider_established = true; sr.nominated_epoch = 5;
        expire_association(sr, GetTickCount64());
        if (!sr.provider_established) return false;
        // Evaluations the API rejects keep the owner only as long as silence
        // would (BF6 10-22: 70 s of mono while every DLSS call was rejected),
        // whatever new calls arrive; an accepted one ends the rejected run,
        // and a late older rejection does not restart it.
        auto &view = evaluations[provider_index(provider_kind::streamline)].views[0];
        const auto finished = [&](std::uint64_t id, std::uint64_t sequence, bool successful) {
          slots[0] = {}; slots[0].id = id;
          slots[0].metadata.provider = provider_kind::streamline; slots[0].metadata.epoch = 5;
          slots[0].metadata.sequence = sequence; slots[0].metadata.viewport = 0; slots[0].metadata.source_id = 0;
          finish(id, successful);
          return slots[0].finished && slots[0].success == successful;
        };
        if (view.sequence != 3 || !finished(41, 3, false) || !view.rejected_since) return false;
        expire_association(sr, GetTickCount64());
        if (!sr.provider_established) return false; // A brief rejected run keeps the owner.
        view.rejected_since -= association_timeout_ms + 1;
        begin_evaluation(5, 4, 0, provider_kind::streamline);
        if (view.sequence != 4 || !view.rejected_since) return false;
        expire_association(sr, GetTickCount64());
        if (sr.provider_established) return false;
        if (!finished(42, 4, true) || view.rejected_since || !finished(43, 3, false) || view.rejected_since) return false;
        sr.provider_established = true; sr.nominated_epoch = 5;
        expire_association(sr, GetTickCount64());
        if (!sr.provider_established) return false;
        fixture::clear_captures();
      }
      {
        // An enabled FG scope owns its viewport only while live: within a
        // second of FG On, or of a recorded copy of its own source that the
        // API accepted (Assassin's Creed Shadows 10-22 kept FG on with no FG
        // depth captured, which locked the viewport's SR depth out).
        constexpr std::uint64_t epoch = 7, fg = (1ull << 63) | 2;
        const auto finished = [&](std::uint64_t id, std::uint64_t source, std::uint32_t viewport, std::uint64_t sequence,
            bool successful, bool metadata_only = false) {
          slots[0] = {}; slots[0].id = id; slots[0].nomination_only = metadata_only;
          slots[0].metadata.provider = provider_kind::streamline; slots[0].metadata.epoch = epoch;
          slots[0].metadata.sequence = sequence; slots[0].metadata.viewport = viewport; slots[0].metadata.source_id = source;
          slots[0].metadata.tick = GetTickCount64();
          finish(id, successful);
          return slots[0].finished;
        };
        const auto silent_since = [] { return GetTickCount64() - association_timeout_ms - 1; };
        if (!frame_generation_scope_live(epoch, 2, GetTickCount64()) || frame_generation_scope_live(epoch, 2, silent_since()) ||
            frame_generation_scope_live(epoch, 2, 0)) return false;
        // A refused FG boundary begins an evaluation but accepts no copy.
        begin_evaluation(epoch, 10, 2, provider_kind::streamline, fg, 0);
        if (frame_generation_scope_live(epoch, 2, silent_since())) return false;
        // A failed evaluation or a metadata-only nomination is no accepted copy.
        if (!finished(51, fg, 2, 10, false) || frame_generation_scope_live(epoch, 2, silent_since())) return false;
        begin_evaluation(epoch, 11, 2, provider_kind::streamline, fg, 0);
        if (!finished(52, fg, 2, 11, true, true) || frame_generation_scope_live(epoch, 2, silent_since())) return false;
        // Ordinary depth of the viewport, or FG of another viewport or epoch,
        // does not make this scope live.
        begin_evaluation(epoch, 12, 2, provider_kind::streamline, 0, UINT64_MAX);
        if (!finished(53, 0, 2, 12, true) || frame_generation_scope_live(epoch, 2, silent_since())) return false;
        begin_evaluation(epoch, 13, 3, provider_kind::streamline, (1ull << 63) | 3, 0);
        if (!finished(54, (1ull << 63) | 3, 3, 13, true) || frame_generation_scope_live(epoch, 2, silent_since()) ||
            frame_generation_scope_live(epoch + 1, 3, silent_since()) || !frame_generation_scope_live(epoch, 3, silent_since()))
          return false;
        // An accepted copy of its own source makes it live for a second.
        begin_evaluation(epoch, 14, 2, provider_kind::streamline, fg, 0);
        if (!finished(55, fg, 2, 14, true) || !frame_generation_scope_live(epoch, 2, silent_since())) return false;
        auto &current = evaluations[provider_index(provider_kind::streamline)];
        const auto own = std::find_if(current.views.begin(), current.views.end(), [&](const evaluation_head &head) {
          return head.epoch == epoch && head.viewport == 2 && head.source_id == fg;
        });
        if (own == current.views.end() || !own->accepted_tick) return false;
        own->accepted_tick -= association_timeout_ms + 1;
        if (frame_generation_scope_live(epoch, 2, silent_since())) return false;
        // Without an active capture owner only the first second counts.
        own->accepted_tick = GetTickCount64(); requested = false;
        if (frame_generation_scope_live(epoch, 2, silent_since()) || !frame_generation_scope_live(epoch, 2, GetTickCount64()))
          return false;
        requested = true; fixture::clear_captures();
      }
      queue_state owner;
      constexpr std::uint64_t native = 77;
      {
        // These prerequisites belong to capture ownership, not the provider's
        // diagnostic consumer. Exercise each refusal without COM/GPU work.
        slot next;
        next.id = 301; next.metadata.provider = provider_kind::ngx;
        next.metadata.epoch = 4; next.metadata.sequence = 201;
        next.metadata.source_id = 7; next.metadata.viewport = 2;
        next.finished = next.success = next.producer_submitted = true;
        next.producer_fence = 12;
        capture_pick picked;
        picked.value = &next; picked.latest = &next;
        picked.result = status::submitted;
        picked.selection = selection_reason::completed_already_consumed;
        const auto classified = [&](const capture_pick &choice) { return classify_acquisition(choice, owner, {}); };
        const auto proof = classified(picked);
        if (!proof.source_selected || !proof.source_valid ||
            proof.provider != next.metadata.provider || proof.capture_id != next.id ||
            proof.epoch != next.metadata.epoch || proof.sequence != next.metadata.sequence ||
            proof.source_id != next.metadata.source_id || proof.viewport != next.metadata.viewport) return false;
        auto no_selection = picked; no_selection.value = nullptr;
        if (classified(no_selection).source_selected) return false;
        owner.last_epoch = next.metadata.epoch; owner.last_sequence = next.metadata.sequence;
        if (!classified(no_selection).repeated_frame) return false;
        no_selection.result = status::failed;
        if (classified(no_selection).repeated_frame) return false;
        next.finished = next.success = false;
        if (!classified(picked).source_valid) return false; // Unfinished is not a failed result.
        next.finished = true;
        if (classified(picked).source_valid) return false;
        capture_pick absent;
        const auto missing = classify_acquisition(absent, owner, {true, 4, 2});
        if (missing.source_selected || missing.source_valid || missing.pending_frame || missing.repeated_frame ||
            missing.epoch != 4 || missing.viewport != 2 || missing.source_id != ((1ull << 63) | 2)) return false;
        owner.last_epoch = owner.last_sequence = 0;
      }
      {
        // The continuity consumer needs a precise refusal reason, not merely
        // "submitted". Exercise the same pre-GPU fallback checks used below.
        auto source = std::make_shared<source_reference>();
        source->desc.Width = 160; source->desc.Height = 90; source->desc.Format = DXGI_FORMAT_R32_FLOAT;
        slot completed, latest;
        completed.id = 100; completed.metadata.epoch = 2; completed.metadata.sequence = 10;
        completed.metadata.source = source;
        completed.metadata.resource.area = {0, 0, 160, 90};
        latest = completed; latest.id = 101; latest.metadata.sequence = 11;
        queue_state consumed;
        consumed.last_epoch = 2; consumed.last_sequence = 10;
        consumed.present = 21; consumed.present_capture = latest.id;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_already_consumed)
          return false;
        // Repeated effects in the same presentation must keep this evidence;
        // an existing pending admission is not a different-source failure.
        consumed.present = 20;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_already_consumed)
          return false;
        // Dynamic resolution: the successor's new allocation (a typed alias of
        // the same family) or new extent is the same source, never a refusal.
        auto resized = std::make_shared<source_reference>();
        resized->desc = source->desc; resized->desc.Width = 176; resized->desc.Height = 99;
        resized->desc.Format = DXGI_FORMAT_D32_FLOAT;
        latest.metadata.source = resized;
        for (const sunshine_scene_depth::extent area : {sunshine_scene_depth::extent{0, 0, 176, 99},
            sunshine_scene_depth::extent{0, 0, 150, 84}, sunshine_scene_depth::extent{8, 4, 160, 90}}) {
          latest.metadata.resource.area = area;
          if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_already_consumed)
            return false;
        }
        // Another storage family, another view of one family (a declared
        // R16_TYPELESS read as FLOAT, then as UNORM) or another resource kind
        // is another encoding.
        for (unsigned change = 0; change != 3; ++change) {
          auto other = std::make_shared<source_reference>();
          other->desc = resized->desc;
          auto before = completed; auto after = latest;
          after.metadata.source = other;
          if (change == 0) other->desc.Format = DXGI_FORMAT_R16_UNORM;
          if (change == 1) {
            auto half = std::make_shared<source_reference>();
            half->desc = source->desc; half->desc.Format = DXGI_FORMAT_R16_TYPELESS;
            before.metadata.source = half; before.metadata.resource.native_format = DXGI_FORMAT_R16_FLOAT;
            other->desc.Format = DXGI_FORMAT_R16_TYPELESS; after.metadata.resource.native_format = DXGI_FORMAT_R16_UNORM;
          }
          if (change == 2) after.metadata.resource.kind = sunshine_scene_depth::resource_kind::display_depth;
          if (completed_fallback_reason(consumed, 21, after, before) != selection_reason::format_changed) return false;
        }
        // The checks below run with the resized successor: it reaches
        // completed_snapshot like an unchanged one.
        consumed.admission_after = completed.id;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_before_gap) return false;
        consumed.admission_after = 0; consumed.last_sequence = 9;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_snapshot) return false;
        completed.metadata.sequence = latest.metadata.sequence;
        if (completed_fallback_reason(consumed, 21, latest, completed) != selection_reason::completed_not_older) return false;
      }
      const auto make = [&](unsigned index, provider_kind provider,
          std::uint64_t epoch, std::uint64_t sequence, std::uint64_t source) -> slot & {
        return fixture::make_capture(index, native, provider, epoch, sequence, source);
      };
      const auto select = [&](std::uint64_t present) { return select_provider(owner, native, present, GetTickCount64(), {}); };
      // An observed API with no usable capture cannot suppress another source.
      begin_evaluation(1, 100, 0, provider_kind::streamline);
      if (owner.provider_established || select(1).value ||
          select(1).selection != selection_reason::no_current_nomination) return false;
      auto &ngx = make(0, provider_kind::ngx, 1, 2, 99);
      ngx.finished = false;
      if (select(1).value || owner.provider_established) return false;
      ngx.finished = true; ngx.success = false;
      if (select(1).value) return false;
      ngx.success = true; ngx.producer_submitted = false;
      if (select(1).value) return false;
      ngx.producer_submitted = true; ngx.queue = native + 1;
      if (select(1).value) return false;
      ngx.queue = native;
      auto chosen = select(1);
      if (chosen.value != &ngx) return false;
      establish_provider(owner, ngx, 1);
      if (!owner.provider_established || owner.provider != provider_kind::ngx || owner.source_id != 99) return false;
      // Acquisition without a successful copy must not consume this frame.
      if (select(2).value != &ngx) return false;
      // complete_frame() bookkeeping for a successful copy.
      owner.last_epoch = ngx.metadata.epoch; owner.last_sequence = ngx.metadata.sequence;
      owner.last_source_tick = ngx.metadata.tick;
      // NGX's already copied frame is not a pending successor, so a ready SL
      // source takes priority on the next presentation; repeated effects
      // within the already-selected presentation remain on NGX.
      make(1, provider_kind::streamline, 1, 101, 0);
      if (select(1).value != &ngx || select(2).value != &slots[1]) return false;
      slots[1].metadata.frame_generation_input = true;
      slots[1].metadata.source_id = 1ull << 63;
      slots[1].metadata.sequence = 102;
      begin_evaluation(1, 102, 0, provider_kind::streamline, slots[1].metadata.source_id, 0);
      if (select(2).value != &slots[1] || select_provider(owner, native, 2, GetTickCount64(), {true, 1, 0}).value != &slots[1]) return false;
      queue_state takeover;
      takeover.provider_established = true; takeover.provider = provider_kind::ngx;
      takeover.source_id = 99; takeover.nominated_epoch = takeover.last_epoch = 1; takeover.last_sequence = 999;
      establish_provider(takeover, slots[1], 2);
      if (select_provider(takeover, native, 3, GetTickCount64(), {}).value != &slots[1]) return false;
      slots[1].metadata.frame_generation_input = false;
      begin_evaluation(0, 9999, UINT32_MAX, provider_kind::streamline);
      if (select(1).value != &ngx) return false;
      begin_evaluation(1, 3, 0, provider_kind::ngx, 99);
      if (select(2).value || !owner.provider_established || owner.source_id != 99) return false;
      auto &resumed = make(2, provider_kind::ngx, 1, 3, 99);
      if (select(2).value != &resumed) return false;
      establish_provider(owner, resumed, 2);
      // Epoch restart may reset sequence; it does not inherit the old watermark.
      auto &recreated = make(3, provider_kind::ngx, 2, 1, 100);
      if (select(3).value != &recreated) return false;
      establish_provider(owner, recreated, 3);
      if (owner.source_id != 100 || owner.provider != provider_kind::ngx) return false;
      // Fresh queues prefer ready SL even when NGX captured first.
      fixture::clear_captures();
      owner.provider_established = false; owner.present_capture = owner.last_sequence = owner.last_epoch = 0;
      make(0, provider_kind::ngx, 3, 1, 123);
      auto &preferred = make(1, provider_kind::streamline, 3, 1, 0);
      if (select(4).value != &preferred) return false;
      // Simultaneous distinct view/source identities remain ambiguous.
      begin_evaluation(3, 2, 1, provider_kind::ngx, 124);
      const auto ambiguous = select_capture(owner, native, 4, provider_kind::ngx, GetTickCount64());
      if (ambiguous.value || ambiguous.result != status::ambiguous ||
          ambiguous.selection != selection_reason::ambiguous_views) return false;
      {
        fixture::clear_captures();
        queue_state preferred_owner;
        auto &fallback = make(0, provider_kind::ngx, 60, 1, 500);
        auto &preferred = make(1, provider_kind::streamline, 70, 1, 0);
        preferred.finished = false;
        const auto pick = [&](std::uint64_t presentation, selection_policy policy = {}) {
          return select_provider(preferred_owner, native, presentation, GetTickCount64(), policy);
        };
        if (pick(1).value != &fallback) return false;
        establish_provider(preferred_owner, fallback, 1);
        preferred_owner.last_epoch = 60; preferred_owner.last_sequence = 1;
        preferred_owner.last_source_tick = fallback.metadata.tick;
        // A submitted nomination without readable pixels must not take over.
        preferred.finished = true; preferred.nomination_only = true;
        preferred.pixel_failure = status::unsupported_resource;
        auto source = std::make_shared<source_reference>(); source->device_identity = 17;
        preferred.metadata.source = source; preferred.source_nominated = true;
        preferred_owner.device_identity = 17;
        if (pick(2).value == &preferred) return false;
        preferred.nomination_only = false; preferred.source_nominated = false;
        // The owner's already copied frame is not a pending successor; it does
        // not hold a readable capture from the other provider.
        if (pick(2).value != &preferred) return false;
        // A live owner that is still delivering keeps its source while its real
        // successor is pending: the other provider does not make selection alternate.
        make(2, provider_kind::ngx, 60, 2, 500).finished = false;
        if (pick(2).value == &preferred) return false;
        // An owner without a recently copied frame holds nothing.
        preferred_owner.last_source_tick = GetTickCount64() - sunshine_scene_depth::maximum_source_age_ms - 1;
        if (pick(2).value != &preferred) return false;
        // Metadata observed after the copy (camera reset, tag change, loss)
        // never makes a readable SL copy yield to a ready NGX source.
        preferred.metadata.observation_revision = 5;
        make(2, provider_kind::ngx, 60, 2, 500);
        queue_state unowned; unowned.device_identity = preferred_owner.device_identity;
        if (select_provider(unowned, native, 2, GetTickCount64(), {}).value != &preferred) return false;
        establish_provider(preferred_owner, preferred, 2);
        preferred_owner.last_epoch = 70; preferred_owner.last_sequence = 1;
        preferred_owner.last_source_tick = preferred.metadata.tick;
        auto &next_fallback = make(2, provider_kind::ngx, 60, 2, 500);
        auto &pending = make(3, provider_kind::streamline, 70, 2, 0); pending.finished = false;
        auto held = pick(3);
        if (held.value || held.latest != &pending || held.result != status::recorded) return false;
        preferred_owner.last_source_tick = GetTickCount64() - sunshine_scene_depth::maximum_source_age_ms - 1;
        if (pick(3).value != &next_fallback) return false;
        preferred_owner.last_source_tick = GetTickCount64();
        pending.finished = true; pending.success = false;
        if (pick(3).value != &next_fallback) return false;
        pending.success = true;
        if (pick(3).value != &pending) return false;
        begin_evaluation(70, 3, 0, provider_kind::streamline);
        if (pick(3).value != &next_fallback) return false;
      }
      const auto empty_queue = std::find_if(queues.begin(), queues.end(), [](const auto &entry) { return !entry; });
      if (empty_queue == queues.end()) return false;
      *empty_queue = std::make_unique<queue_state>();
      struct release_queue { std::unique_ptr<queue_state> &entry; ~release_queue() { entry.reset(); } } queue_cleanup{*empty_queue};
      auto &established = **empty_queue;
      {
        // An observed FG mode is source authority even before SL has usable
        // pixels. Older NGX, ordinary SL SR, and a different viewport/epoch
        // must not sneak through that gap or supply its camera separately.
        fixture::clear_captures();
        constexpr std::uint64_t fg_epoch = 42, fg_id = 1ull << 63;
        const selection_policy fg_policy{true, fg_epoch, 0};
        const auto pick = [&](std::uint64_t present, selection_policy policy = {}) {
          return select_provider(established, native, present, GetTickCount64(), policy);
        };
        auto &first_ngx = make(0, provider_kind::ngx, 30, 1, 400);
        if (pick(20).value != &first_ngx || pick(20, fg_policy).value) return false;
        establish_provider(established, first_ngx, 20);
        // Turning FG off again before its first admitted pixels must leave
        // the established ordinary source usable; intent does not own it.
        if (pick(21, fg_policy).value || pick(21).value != &first_ngx) return false;
        auto &sr = make(1, provider_kind::streamline, fg_epoch, 1, 0);
        sr.metadata.projection = {1, -1, true, false};
        if (pick(21, fg_policy).value) return false;
        auto &fg = make(2, provider_kind::streamline, fg_epoch, 2, fg_id);
        fg.metadata.frame_generation_input = true;
        fg.metadata.projection = {0, 1, true, true};
        fg.metadata.sequence = 3;
        begin_evaluation(fg_epoch, 3, 0, provider_kind::streamline, fg_id, 0);
        fg.finished = false;
        auto pending = pick(21, fg_policy);
        if (pending.value || pending.latest != &fg || pending.result != status::recorded ||
            established.provider != provider_kind::ngx) return false;
        fg.finished = true;
        const auto accepted = pick(21, fg_policy);
        if (accepted.value != &fg || accepted.value->metadata.projection.depth_offset != 0 ||
            accepted.value->metadata.projection.depth_scale != 1 ||
            pick(21, {true, fg_epoch + 1, 0}).value || pick(21, {true, fg_epoch, 1}).value) return false;
        // No prior provider: the same policy still beats the earlier NGX ID.
        queue_state fresh;
        if (select_provider(fresh, native, 21, GetTickCount64(), fg_policy).value != &fg) return false;
        queue_state sr_owner;
        establish_provider(sr_owner, sr, 20);
        sr_owner.last_epoch = fg_epoch; sr_owner.last_sequence = 1;
        if (select_provider(sr_owner, native, 21, GetTickCount64(), fg_policy).value != &fg) return false;
        // A transient missing camera does not invent one from NGX or revoke
        // otherwise valid SL depth; consumers retain the raw-depth path.
        fg.metadata.projection.supplied = false;
        if (pick(21, fg_policy).value != &fg || pick(21, fg_policy).value->metadata.projection.supplied) return false;
        fg.metadata.projection.supplied = true;
        establish_provider(established, fg, 21);
        established.last_epoch = fg_epoch; established.last_sequence = 3;
        begin_evaluation(fg_epoch, 4, 0, provider_kind::streamline, fg_id);
        make(3, provider_kind::ngx, 30, 2, 400);
        if (pick(22, fg_policy).value) return false;
        // A successful FG Off retires that exact source; subsequent normal
        // acquisition resumes a new NGX frame, never reviving pre-gap pixels.
        retire_source(provider_kind::streamline, fg_epoch, fg_id);
        if (established.provider_established || pick(23).value || pick(23, fg_policy).value) return false;
        auto &resumed_ngx = make(4, provider_kind::ngx, 30, 3, 400);
        if (pick(24).value != &resumed_ngx || pick(24, fg_policy).value) return false;
      }
      {
        // A failed/ambiguous FG option revokes FG authority without starving
        // independent ordinary depth on subsequent presentations.
        fixture::clear_captures();
        queue_state revoked_owner;
        constexpr std::uint64_t fg_epoch = 42, fg_id = 1ull << 63;
        selection_policy excluded;
        excluded.exclude_unconfirmed_fg = true;
        const auto pick = [&](std::uint64_t present, selection_policy policy) {
          return select_provider(revoked_owner, native, present, GetTickCount64(), policy);
        };
        auto &old_fg = make(0, provider_kind::streamline, fg_epoch, 1, fg_id);
        old_fg.metadata.frame_generation_input = true;
        establish_provider(revoked_owner, old_fg, 30);
        revoked_owner.last_epoch = fg_epoch; revoked_owner.last_sequence = 1;
        revoked_owner.last_source_tick = old_fg.metadata.tick;
        make(1, provider_kind::ngx, 60, 1, 500);
        if (pick(30, excluded).value) return false; // Keep the existing presentation pin.
        revoked_owner.admission_after = serial.load(); // acquire's one-time gap watermark.
        auto &fallback = make(2, provider_kind::ngx, 60, 2, 500);
        fallback.metadata.projection.encoding = sunshine_scene_depth::depth_encoding::linear_distance;
        fallback.metadata.projection.raw_scale = 2;
        for (const auto present : {31ull, 32ull}) {
          const auto selected = pick(present, excluded);
          if (selected.value != &fallback || selected.result != status::ready ||
              selected.value->metadata.projection.encoding != sunshine_scene_depth::depth_encoding::linear_distance ||
              selected.value->metadata.projection.raw_scale != 2) return false;
        }
        fallback.success = false;
        const auto unavailable = pick(31, excluded);
        // The live failed NGX attempt is reported in front of the excluded FG
        // scope. It supplies no source and never reveals the excluded FG input.
        if (unavailable.value || unavailable.pending_nomination || unavailable.latest != &fallback ||
            unavailable.result != status::failed) return false;
        fallback.success = true;
        auto &pending_fg = make(3, provider_kind::streamline, fg_epoch, 2, fg_id);
        pending_fg.metadata.frame_generation_input = true; pending_fg.finished = false;
        if (pick(31, excluded).value != &fallback || pick(31, {true, fg_epoch, 0}).value == &fallback) return false;
        // Retirement invalidates old pixels and removes that FG view, without
        // suppressing a later independent ordinary SL source on this viewport.
        retire_source(provider_kind::streamline, fg_epoch, fg_id);
        if (!old_fg.invalid || !pending_fg.invalid || pick(32, {true, fg_epoch, 0}).value) return false;
        auto &ordinary_sl = make(4, provider_kind::streamline, fg_epoch, 3, 0);
        if (pick(31, excluded).value != &ordinary_sl) return false;
      }
      {
        // A failed options call revokes all captures in its exact FG scope,
        // including tags observed while the SDK call was in flight. A racing
        // newer capture may be discarded once; only a fresh tag restores FG.
        fixture::clear_captures();
        constexpr std::uint64_t fg_epoch = 80, fg_id = 1ull << 63;
        auto &old_fg = make(0, provider_kind::streamline, fg_epoch, 10, fg_id);
        old_fg.metadata.frame_generation_input = true;
        auto &new_fg = make(1, provider_kind::streamline, fg_epoch, 12, fg_id);
        new_fg.metadata.frame_generation_input = true;
        establish_provider(established, new_fg, 100);
        established.last_epoch = fg_epoch; established.last_sequence = 12;
        retire_source(provider_kind::streamline, fg_epoch, fg_id);
        finish(new_fg.id, true);
        if (!old_fg.invalid || !new_fg.invalid || established.provider_established ||
            select_provider(established, native, 101, GetTickCount64(), {true, fg_epoch, 0}).value) return false;
        auto &fresh_ngx = make(2, provider_kind::ngx, 81, 1, 600);
        if (select_provider(established, native, 102, GetTickCount64(), {}).value != &fresh_ngx) return false;
        establish_provider(established, fresh_ngx, 102);
        if (select_provider(established, native, 103, GetTickCount64(), {true, fg_epoch, 0}).value) return false;
        auto &renewed_fg = make(3, provider_kind::streamline, fg_epoch, 14, fg_id);
        renewed_fg.metadata.frame_generation_input = true;
        if (select_provider(established, native, 103, GetTickCount64(), {true, fg_epoch, 0}).value != &renewed_fg) return false;
      }
      for (const bool complete_before_release : {false, true}) {
        fixture::clear_captures();
        auto &retired = make(0, provider_kind::ngx, 4, 1, 200);
        auto &other = make(1, provider_kind::streamline, 4, 1, 0);
        retired.command = 888; retired.retirements[0] = {native, 9};
        retired.finished = complete_before_release;
        establish_provider(established, retired, 5);
        retire_source(provider_kind::ngx, 4, 201);
        retire_source(provider_kind::ngx, 5, 200);
        retire_source(provider_kind::streamline, 4, 200);
        if (retired.invalid || !established.provider_established || established.source_id != 200 || other.invalid) return false;
        retire_source(provider_kind::ngx, 4, 200);
        if (established.provider_established || established.present_capture || established.source_id ||
            !retired.invalid || retired.failure != capture_failure::source_retired ||
            retired.command != 888 || retired.producer_fence != 1 || retired.retirements[0].value != 9 ||
            !retired.producer_submitted || other.invalid || established.admission_after != other.id) return false;
        // A result returning after explicit release cannot revive old pixels.
        finish(retired.id, true);
        if (!retired.invalid || submitted_success(retired, native) || reclaimable(retired)) return false;
        // The immediately preceding capture is exactly serial, so subtracting
        // one at release would wrongly resurrect this other provider's frame.
        if (select_provider(established, native, 6, GetTickCount64(), {}).value) return false;
        auto &fresh = make(2, provider_kind::streamline, 4, 2, 0);
        if (fresh.id <= established.admission_after ||
            select_provider(established, native, 6, GetTickCount64(), {}).value != &fresh) return false;
      }
      {
        // depth_choice (docs/reshade-sbs.md, Frame record) repeats what each
        // pick of this timeline chose and why: its decision, selection reason,
        // status and failure, the role of its source, the policy, and the
        // queue's owner after it. It decides nothing.
        fixture::clear_captures();
        using sunshine_game3d::source_slot;
        queue_state chooser;
        const auto chosen = [&](std::uint64_t present, selection_policy policy = {}) {
          const auto pick = select_provider(chooser, native, present, GetTickCount64(), policy);
          const auto decision = classify_acquisition(pick, chooser, policy);
          const auto out = describe_choice(pick, decision, chooser, policy, present, 1234, pick.result == status::ready);
          const auto *source = pick.value ? pick.value : pick.latest;
          const bool mirrors = out.present == present && out.tick_ms == 1234 && out.result == pick.result &&
            out.selection == pick.selection && out.pixel_ready == (pick.result == status::ready) &&
            out.failure == (source ? source->failure : capture_failure::none) &&
            out.decision.source_selected == decision.source_selected && out.decision.provider == decision.provider &&
            out.decision.epoch == decision.epoch && out.decision.sequence == decision.sequence &&
            out.decision.source_id == decision.source_id && out.decision.capture_id == decision.capture_id &&
            out.decision.viewport == decision.viewport && out.decision.pending_frame == decision.pending_frame &&
            out.decision.repeated_frame == decision.repeated_frame &&
            out.fg_required == policy.require_frame_generation && out.fg_excluded == policy.exclude_unconfirmed_fg &&
            out.owner_established == chooser.provider_established && out.owner == chooser.provider &&
            out.owner_source == chooser.source_id;
          return std::pair{out, mirrors};
        };
        constexpr std::uint64_t fg_epoch = 50, fg_id = 1ull << 63;
        // No API reported anything: no rung.
        const auto [nothing, nothing_mirrors] = chosen(40);
        if (!nothing_mirrors || nothing.rung != source_slot::none || nothing.decision.source_selected) return false;
        // Readable NGX depth is scene depth (D2), and then the queue's owner.
        auto &scene_source = make(0, provider_kind::ngx, 60, 1, 500);
        const auto [scene, scene_mirrors] = chosen(41);
        if (!scene_mirrors || scene.rung != source_slot::scene_depth || !scene.decision.source_selected ||
            scene.decision.capture_id != scene_source.id || scene.result != status::ready || !scene.pixel_ready ||
            scene.owner_established) return false;
        establish_provider(chooser, scene_source, 41);
        const auto [owned, owned_mirrors] = chosen(42);
        if (!owned_mirrors || !owned.owner_established || owned.owner != provider_kind::ngx || owned.owner_source != 500)
          return false;
        // Required FG before its scope is known: the FG rung, no source.
        const auto [missing, missing_mirrors] = chosen(43, {true, 0, 0});
        if (!missing_mirrors || missing.rung != source_slot::fg_depth ||
            missing.selection != selection_reason::fg_scope_missing || missing.decision.source_selected) return false;
        // Required FG with its own input still recording: the FG rung, pending.
        auto &fg_source = make(1, provider_kind::streamline, fg_epoch, 2, fg_id);
        fg_source.metadata.frame_generation_input = true; fg_source.finished = false;
        const auto [pending_fg, pending_mirrors] = chosen(44, {true, fg_epoch, 0});
        if (!pending_mirrors || pending_fg.rung != source_slot::fg_depth || pending_fg.decision.source_selected ||
            pending_fg.result != status::recorded || pending_fg.pixel_ready || !pending_fg.fg_required) return false;
        // Another viewport's scope does not match its input.
        const auto [mismatch, mismatch_mirrors] = chosen(44, {true, fg_epoch, 1});
        if (!mismatch_mirrors || mismatch.rung != source_slot::fg_depth ||
            mismatch.selection != selection_reason::fg_scope_mismatch || mismatch.decision.source_selected) return false;
        // An unconfirmed scope excludes FG input; scene depth still supplies.
        selection_policy excluded;
        excluded.exclude_unconfirmed_fg = true;
        const auto [unconfirmed, unconfirmed_mirrors] = chosen(45, excluded);
        if (!unconfirmed_mirrors || unconfirmed.rung != source_slot::scene_depth || !unconfirmed.fg_excluded ||
            unconfirmed.decision.capture_id != scene_source.id) return false;
        // Its finished input is FG depth (D1), ready.
        fg_source.finished = true;
        const auto [fg_ready, fg_mirrors] = chosen(46, {true, fg_epoch, 0});
        if (!fg_mirrors || fg_ready.rung != source_slot::fg_depth || !fg_ready.decision.source_selected ||
            fg_ready.decision.capture_id != fg_source.id || !fg_ready.pixel_ready) return false;
        fixture::clear_captures();
      }
      // SR and FG use distinct logical sources for the same viewport. The
      // explicit role transition removes only the superseded SR view, rather
      // than waiting for its normal age window or ignoring true multi-view.
      fixture::clear_captures();
      queue_state role_owner;
      constexpr std::uint64_t fg_source = 1ull << 63;
      make(0, provider_kind::streamline, 8, 1, 0);
      auto &fg = make(1, provider_kind::streamline, 8, 2, fg_source);
      if (select_capture(role_owner, native, 7, provider_kind::streamline, GetTickCount64()).result != status::ambiguous) return false;
      fg.metadata.sequence = 3;
      begin_evaluation(8, 3, 0, provider_kind::streamline, fg_source, 0);
      if (select_capture(role_owner, native, 7, provider_kind::streamline, GetTickCount64()).value != &fg) return false;
      begin_evaluation(8, 4, 1, provider_kind::streamline, 0);
      fg.metadata.sequence = 5;
      begin_evaluation(8, 5, 0, provider_kind::streamline, fg_source, 0);
      if (select_capture(role_owner, native, 7, provider_kind::streamline, GetTickCount64()).result != status::ambiguous) return false;

      // A Present can observe an SDK nomination after its watermark advances
      // but before native validation/allocation has produced a capture slot.
      // Such an intent can retain only an established complete FG pair, never
      // select the previous depth or establish authority on a fresh queue.
      queue_state pending_owner;
      const auto prepare_pending = [&] {
        fixture::clear_captures();
        pending_owner.provider_established = false;
        pending_owner.last_epoch = pending_owner.last_sequence = pending_owner.present = pending_owner.present_capture = 0;
        auto &previous = make(0, provider_kind::streamline, 20, 1, fg_source);
        previous.metadata.frame_generation_input = true;
        establish_provider(pending_owner, previous, 10);
        pending_owner.last_epoch = 20; pending_owner.last_sequence = 1;
        input candidate = previous.metadata;
        candidate.sequence = 2;
        candidate.source = std::make_shared<source_reference>();
        candidate.resource.native = 1;
        return candidate;
      };
      const auto pending_pick = [&](std::uint64_t now = GetTickCount64()) {
        return select_capture(pending_owner, native, 11, provider_kind::streamline, now);
      };
      auto candidate = prepare_pending();
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      auto pending = pending_pick();
      if (pending.value || pending.latest || !pending.pending_nomination || pending.result != status::recorded ||
          !matching_evaluation(*pending.pending_nomination, candidate)) return false;
      auto &pending_fallback = make(1, provider_kind::ngx, 60, 1, 500);
      selection_policy excluded_pending;
      excluded_pending.exclude_unconfirmed_fg = true;
      if (select_provider(pending_owner, native, 11, GetTickCount64(), excluded_pending).value != &pending_fallback) return false;
      const auto pending_decision = classify_acquisition(pending, pending_owner, {true, 20, 0});
      if (pending_decision.source_selected || !pending_decision.source_valid || !pending_decision.pending_frame ||
          pending_decision.capture_id || pending_decision.epoch != candidate.epoch ||
          pending_decision.sequence != candidate.sequence || pending_decision.source_id != candidate.source_id ||
          pending_decision.viewport != candidate.viewport) return false;
      pending_owner.provider_established = false;
      if (pending_pick().pending_nomination || pending_pick().value) return false;
      pending_owner.provider_established = true;
      if (pending_pick(pending.pending_nomination->tick + 251).pending_nomination ||
          pending_pick(pending.pending_nomination->tick + 251).result != status::stale) return false;
      end_nomination(candidate);
      if (pending_pick().pending_nomination || pending_pick().value) return false;

      // An old call returning cannot clear a later valid intent, and a missing
      // newer tag withdraws it even though no record()/finish() follows.
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      auto newer = candidate; ++newer.sequence;
      if (!begin_nomination(newer, UINT64_MAX)) return false;
      end_nomination(candidate);
      if (!pending_pick().pending_nomination || !matching_evaluation(*pending_pick().pending_nomination, newer)) return false;
      begin_evaluation(20, 4, 0, provider_kind::streamline, fg_source);
      end_nomination(newer);
      if (pending_pick().pending_nomination || pending_pick().value || pending_pick().latest) return false;

      // Exercise the real early-rejection/RAII path without fake native COM.
      candidate = prepare_pending();
      record_diagnostic rejected;
      if (nominate_evaluation(0, candidate, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
          pending_pick().pending_nomination || pending_pick().value) return false;
      input alternatives[]{candidate, candidate, candidate};
      if (nominate_evaluation(0, alternatives, 2, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
          pending_pick().pending_nomination || pending_pick().value) return false;
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      if (nominate_evaluation(0, alternatives, 3, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
          pending_pick().pending_nomination || pending_pick().value) return false;
      // A batch cannot combine logical evaluations or revoke a separately
      // active nomination when its adapter contract is malformed.
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      for (unsigned mismatch = 0; mismatch != 6; ++mismatch) {
        alternatives[1] = candidate;
        if (mismatch == 0) alternatives[1].provider = provider_kind::ngx;
        if (mismatch == 1) ++alternatives[1].epoch;
        if (mismatch == 2) ++alternatives[1].sequence;
        if (mismatch == 3) ++alternatives[1].source_id;
        if (mismatch == 4) ++alternatives[1].viewport;
        if (mismatch == 5) alternatives[1].frame_generation_input = false;
        if (nominate_evaluation(0, alternatives, 2, UINT64_MAX, &rejected) || rejected.stage != record_stage::malformed_input ||
            !pending_pick().pending_nomination) return false;
      }
      if (nominate_evaluation(0, nullptr, 1, UINT64_MAX, &rejected) ||
          nominate_evaluation(0, alternatives, 0, UINT64_MAX, &rejected) ||
          nominate_evaluation(0, alternatives, 4, UINT64_MAX, &rejected) || !pending_pick().pending_nomination) return false;
      end_nomination(candidate);
      for (unsigned mismatch = 0; mismatch != 4; ++mismatch) {
        candidate = prepare_pending();
        if (mismatch == 0) ++candidate.epoch;
        if (mismatch == 1) ++candidate.source_id;
        if (mismatch == 2) ++candidate.viewport;
        if (!begin_nomination(candidate, UINT64_MAX)) return false;
        auto &current = evaluations[provider_index(candidate.provider)];
        if (mismatch == 2) current.views[0] = {}; // Isolate exact viewport identity from ambiguity.
        if (mismatch == 3) current.views[1] = {20, 2, GetTickCount64(), fg_source + 1, 1};
        if (pending_pick().pending_nomination || pending_pick().value) return false;
        end_nomination(candidate);
      }
      candidate = prepare_pending();
      if (!begin_nomination(candidate, UINT64_MAX)) return false;
      auto &published = slots[1];
      published.id = ++serial; published.metadata = candidate;
      published.source_nominated = published.nomination_only = true;
      published.pixel_failure = status::unsupported_resource;
      if (pending_pick().latest || !pending_pick().pending_nomination ||
          !pending_frame(pending_pick(), pending_owner) || pending_pick().value) return false;
      // A rejected intermediate tag does not end the enclosing batch: the next
      // API-supplied tag can still capture. Ending that scope removes the hold.
      finish(published.id, false);
      if (pending_pick().latest || pending_pick().result != status::recorded ||
          !pending_frame(pending_pick(), pending_owner)) return false;
      end_nomination(candidate);
      if (pending_pick().result != status::failed || pending_frame(pending_pick(), pending_owner)) return false;
      // A successfully tagged but unsupported texture has no pending pixels,
      // even before producer submission. It must use current-color mono.
      published = {}; published.id = ++serial; published.metadata = candidate;
      published.source_nominated = published.nomination_only = published.finished = published.success = true;
      published.pixel_failure = status::unsupported_resource;
      if (pending_pick().result != status::recorded || pending_frame(pending_pick(), pending_owner)) return false;
      published.nomination_only = false;
      if (!pending_frame(pending_pick(), pending_owner)) return false; // A real unfinished snapshot may hold.
      published.success = false;
      if (pending_frame(pending_pick(), pending_owner)) return false;

      // Preferred metadata must use its reserved capacity before taking a
      // pixel-capable slot needed by a lower-priority tag's actual capture.
      fixture::clear_captures();
      begin_evaluation(candidate.epoch, candidate.sequence, candidate.viewport, candidate.provider, candidate.source_id);
      for (unsigned i = 0; i + 1 < pixel_slot_limit; ++i) {
        slots[i].id = ++serial;
        slots[i].retirement_unknown = true; // Live GPU obligations cannot be reclaimed.
      }
      const auto pixel_end = slots.begin() + pixel_slot_limit;
      for (unsigned i = pixel_slot_limit; i < slot_limit; ++i) {
        const auto ticket = record_nomination(candidate, candidate.resource.native, 0, {});
        if (!ticket || slots[i].id != ticket ||
            std::find_if(slots.begin(), pixel_end, reclaimable) != pixel_end - 1) return false;
      }
      // The reservation is a preference, not an artificial nomination limit:
      // when it is full, unused pixel capacity can still retain source authority.
      const auto overflow = record_nomination(candidate, candidate.resource.native, 0, {});
      if (!overflow || slots[pixel_slot_limit - 1].id != overflow ||
          record_nomination(candidate, candidate.resource.native, 0, {})) return false;
      finish(slots[pixel_slot_limit].id, false);
      const auto recycled = record_nomination(candidate, candidate.resource.native, 0, {});
      if (!recycled || slots[pixel_slot_limit].id != recycled ||
          slots[pixel_slot_limit - 1].id != overflow) return false;
      return true;
    }
    bool live_source_admission_regression() {
      // A pipelined game's newest nomination is still in flight at every
      // Present, so its pixels come only from the source's own newest
      // completed snapshot. Drive the production selector together with the
      // watermark, establishment and consumption bookkeeping of acquire() and
      // complete_frame(); only the producer fences are supplied by the CPU.
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        std::array<evaluation_namespace, provider_count> saved_evaluations{evaluations};
        std::array<status, provider_count> saved_attempts{attempt_status[0].load(), attempt_status[1].load()};
        bool saved_requested{requested.load()};
        status saved_reason{reason.load()};
        std::uint64_t saved_serial{serial.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = saved_attempts[i];
          requested = saved_requested; reason = saved_reason; serial = saved_serial;
        }
      } restore;
      requested = true;
      struct completion_fence final : ID3D12Fence {
        UINT64 completed{};
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void **out) override { if (out) *out = nullptr; return E_NOINTERFACE; }
        ULONG STDMETHODCALLTYPE AddRef() override { return 1; }
        ULONG STDMETHODCALLTYPE Release() override { return 1; }
        HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT *, void *) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void *) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown *) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE GetDevice(REFIID, void **out) override { if (out) *out = nullptr; return E_NOINTERFACE; }
        UINT64 STDMETHODCALLTYPE GetCompletedValue() override { return completed; }
        HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64, HANDLE) override { return E_NOTIMPL; }
        HRESULT STDMETHODCALLTYPE Signal(UINT64) override { return E_NOTIMPL; }
      };
      // Registered queue timelines. Selection only compares a queue's identity
      // and reads its private fence; it never calls the queue.
      struct producer_queue {
        completion_fence fence;
        std::uint64_t signaled{};
        std::unique_ptr<queue_state> *entry{};
        std::uint64_t native() const { return reinterpret_cast<std::uint64_t>(this); }
        ~producer_queue() {
          if (!entry || !*entry) return;
          (*entry)->queue.p = nullptr; (*entry)->fence.p = nullptr; entry->reset();
        }
      };
      constexpr std::uint64_t device = 41;
      // Two producer timelines (SL, NGX) and the effects queue, which owns the
      // provider selection exactly as in acquire(), so retire_source sees it.
      std::array<producer_queue, 3> producers;
      for (auto &value : producers) {
        const auto entry = std::find_if(queues.begin(), queues.end(), [](const auto &item) { return !item; });
        if (entry == queues.end()) return false;
        *entry = std::make_unique<queue_state>();
        value.entry = &*entry;
        (*entry)->queue.p = reinterpret_cast<ID3D12CommandQueue *>(&value);
        (*entry)->fence.p = &value.fence;
        (*entry)->device_identity = device;
      }
      const auto consumer = producers[2].native();
      const auto fresh_owner = [&]() -> queue_state & {
        auto &value = **producers[2].entry;
        value.provider_established = false; value.provider = provider_kind::streamline;
        value.source_id = value.last_epoch = value.last_sequence = value.last_source_tick = value.nominated_epoch = 0;
        value.present = value.present_capture = value.admission_after = 0; value.viewport = 0;
        return value;
      };
      const auto layout = [&](unsigned width, unsigned height) {
        auto value = std::make_shared<source_reference>();
        value->device_identity = device;
        value->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        value->desc.Width = width; value->desc.Height = height; value->desc.Format = DXGI_FORMAT_R32_TYPELESS;
        return value;
      };
      struct pipeline {
        provider_kind provider{};
        std::uint64_t epoch{}, source_id{};
        source_ref source;
        producer_queue *queue{};
        unsigned first{};
        std::uint64_t sequence{};
        bool frame_generation{};
        unsigned written{};
        slot *pending{};
      };
      // The game records its next evaluation, still in flight at the next
      // Present, while its previous one completes on the GPU.
      const auto advance = [&](pipeline &p) -> slot & {
        if (p.pending) p.queue->fence.completed = std::max<UINT64>(p.queue->fence.completed, p.pending->producer_fence);
        auto &value = slots[p.first + p.written++ % 4];
        value = {};
        value.id = ++serial; value.queue = p.queue->native(); value.producer_fence = ++p.queue->signaled;
        value.texture = std::make_shared<texture_reference>();
        value.finished = value.success = value.producer_submitted = value.source_nominated = true;
        value.metadata.provider = p.provider; value.metadata.epoch = p.epoch; value.metadata.sequence = ++p.sequence;
        value.metadata.source_id = p.source_id; value.metadata.source = p.source;
        value.metadata.frame_generation_input = p.frame_generation;
        value.metadata.tick = GetTickCount64();
        begin_evaluation(p.epoch, p.sequence, 0, p.provider, p.source_id);
        p.pending = &value;
        return value;
      };
      const auto finish_gpu = [](pipeline &p) {
        p.queue->fence.completed = std::max<UINT64>(p.queue->fence.completed, p.pending->producer_fence);
      };
      // An attempt rejected before its capture slot exists (observer coverage,
      // a closed recording or render pass, proof, lifetime or identity checks):
      // the provider's sequence and view advance, its previous capture
      // completes, and only the attempt status remains.
      const auto reject_without_slot = [&](pipeline &p) {
        if (p.pending) finish_gpu(p);
        begin_evaluation(p.epoch, ++p.sequence, 0, p.provider, p.source_id);
        attempt_status[provider_index(p.provider)] = status::unsupported_state;
      };
      // Time passes for a source that stopped evaluating, including the last
      // frame the queue copied from it.
      const auto expire = [&](const pipeline &p, queue_state &queue) {
        constexpr auto age = sunshine_scene_depth::maximum_source_age_ms + 1;
        for (auto &view : evaluations[provider_index(p.provider)].views)
          if (view.source_id == p.source_id && view.tick > age) view.tick -= age;
        for (auto &value : slots)
          if (value.id && value.metadata.provider == p.provider && value.metadata.source_id == p.source_id &&
              value.metadata.tick > age) value.metadata.tick -= age;
        if (queue.provider == p.provider && queue.source_id == p.source_id && queue.last_source_tick > age)
          queue.last_source_tick -= age;
      };
      // The provider's view of each Present: acquire() classifies the pick
      // before establishment, and only a successful copy replaces the
      // retained display frame. Test snapshots own no resource, so packets
      // are described through their source allocation, as nominations are.
      acquisition_decision decided;
      retained_depth retained;
      const auto describe = [&](const slot &value, status result) {
        auto view = value; view.nomination_only = true;
        packet out; describe_packet(view, consumer, out);
        out.pixel_ready = !value.nomination_only && result == status::ready;
        return out;
      };
      // acquire() followed, for ready pixels, by a successful copy and
      // complete_frame(): the same watermark, establishment and consumption.
      // copied=false models a failed display preparation or copy: no complete_frame().
      const auto present_frame = [&](queue_state &queue, std::uint64_t present, selection_policy policy = {},
          bool copied = true) {
        const auto chosen = select_provider(queue, consumer, present, GetTickCount64(), policy);
        decided = classify_acquisition(chosen, queue, policy);
        if (queue.provider_established && chosen.result != status::ready && chosen.result != status::recorded &&
            chosen.result != status::submitted) queue.admission_after = serial.load();
        if (chosen.value) {
          establish_provider(queue, *chosen.value, present);
          if (chosen.result == status::ready && copied) {
            queue.last_epoch = chosen.value->metadata.epoch;
            queue.last_sequence = chosen.value->metadata.sequence;
            queue.last_source_tick = chosen.value->metadata.tick;
            const auto copy = describe(*chosen.value, chosen.result);
            retained = {copy.metadata, copy.capture_id, present, copy.width, copy.height, copy.format, copy.srv_format,
              copy.area};
          }
        }
        return chosen;
      };
      // decide_display() for the last Present against the retained frame.
      const auto display = [&](const capture_pick &chosen) {
        const auto candidate = chosen.value ? describe(*chosen.value, chosen.result) : packet{};
        return decide_display(decided, candidate, retained, {}, GetTickCount64());
      };
      const auto pending_result = [](const capture_pick &value) {
        return value.result == status::recorded || value.result == status::submitted;
      };
      const auto completed_pick = [](const capture_pick &value, const slot *expected) {
        return value.value == expected && value.result == status::ready && value.selection == selection_reason::completed_snapshot;
      };
      // One add-on epoch can number both providers; their sequences are
      // separate namespaces and must never be compared with each other.
      constexpr std::uint64_t epoch = 4;
      const auto render = layout(1280, 720);
      const auto streamline = [&](std::uint64_t source_id = 0) {
        return pipeline{provider_kind::streamline, epoch, source_id, render, &producers[0], 0, 31635};
      };
      const auto ngx = [&](std::uint64_t source_id = 9, source_ref source = {}) {
        return pipeline{provider_kind::ngx, epoch, source_id, source ? source : render, &producers[1], 8, 4686};
      };
      const auto restart = [&] {
        fixture::clear_captures(); decided = {}; retained = {};
        for (auto &value : producers) { value.fence.completed = value.signaled = 0; }
      };
      // Establish a pipelined source as the delivering owner.
      const auto deliver = [&](queue_state &queue, pipeline &p, std::uint64_t &present) {
        advance(p);
        for (unsigned frame = 0; frame != 3; ++frame) {
          auto &completed = *p.pending;
          advance(p);
          if (!completed_pick(present_frame(queue, ++present), &completed)) return false;
        }
        return queue.provider_established && queue.provider == p.provider && queue.source_id == p.source_id;
      };
      std::uint64_t present = 0;

      // (a)+(f) A late SL capture took ownership and SL then stopped. Its
      // expired source must neither hide pipelined NGX nor mark a gap.
      {
        restart();
        auto &owner = fresh_owner();
        auto sl = streamline();
        auto ngx_source = ngx();
        auto &late = advance(sl); finish_gpu(sl);
        if (present_frame(owner, ++present).value != &late || owner.provider != provider_kind::streamline ||
            owner.last_sequence != late.metadata.sequence) return false;
        expire(sl, owner);
        advance(ngx_source);
        const auto first = present_frame(owner, ++present);
        if (first.latest != ngx_source.pending || !pending_result(first) || owner.admission_after) return false;
        auto &completed = *ngx_source.pending;
        advance(ngx_source);
        if (!completed_pick(present_frame(owner, ++present), &completed) || owner.provider != provider_kind::ngx) return false;
        for (unsigned frame = 0; frame != 4; ++frame) {
          auto &next = *ngx_source.pending;
          advance(ngx_source);
          if (!completed_pick(present_frame(owner, ++present), &next)) return false;
        }
        // Without an owner, the expired SL capture is still not reported in
        // front of a live NGX source that has no completed snapshot yet.
        restart();
        auto &fresh = fresh_owner();
        auto stale = streamline(); auto pending = ngx();
        advance(stale); finish_gpu(stale);
        expire(stale, fresh);
        advance(pending).producer_submitted = false; // CPU-recorded only: no source value yet.
        const auto front = select_provider(fresh, consumer, ++present, GetTickCount64(), {});
        if (front.value || front.latest != pending.pending || front.result != status::recorded) return false;
      }
      // (b) No owner (runtime reset or new generation): a source that is only
      // ever CPU-recorded at Present establishes from its completed snapshot.
      {
        restart();
        auto &owner = fresh_owner();
        auto source = ngx();
        advance(source).producer_submitted = false;
        const auto recorded = present_frame(owner, ++present);
        if (recorded.value || recorded.result != status::recorded || owner.provider_established) return false;
        auto &completed = *source.pending; completed.producer_submitted = true;
        advance(source).producer_submitted = false;
        if (!completed_pick(present_frame(owner, ++present), &completed) || !owner.provider_established ||
            owner.provider != provider_kind::ngx || owner.source_id != 9) return false;
      }
      // (c) A DLSS quality change re-creates the NGX feature at a new render
      // resolution: a new source identity whose newest capture is in flight,
      // already submitted or still only CPU-recorded at each Present.
      for (const bool observed_release : {true, false}) for (const bool submitted_newest : {true, false}) {
        restart();
        auto &owner = fresh_owner();
        auto old_feature = ngx(9);
        if (!deliver(owner, old_feature, present)) return false;
        if (observed_release) {
          retire_source(provider_kind::ngx, epoch, 9);
          if (owner.provider_established) return false;
        } else expire(old_feature, owner); // The settings change outlasts the source age.
        auto feature = ngx(10, layout(1706, 960));
        feature.sequence = old_feature.sequence; feature.first = 12;
        advance(feature).producer_submitted = submitted_newest;
        const auto changing = present_frame(owner, ++present);
        if ((changing.value && changing.value->metadata.source_id != 10) || changing.latest != feature.pending ||
            !pending_result(changing)) return false;
        auto &completed = *feature.pending; completed.producer_submitted = true;
        advance(feature).producer_submitted = submitted_newest;
        if (!completed_pick(present_frame(owner, ++present), &completed) || owner.source_id != 10 ||
            owner.provider != provider_kind::ngx) return false;
      }
      // (d) The mirror case: NGX stops and pipelined SL takes over.
      {
        restart();
        auto &owner = fresh_owner();
        auto old_source = ngx();
        if (!deliver(owner, old_source, present)) return false;
        expire(old_source, owner);
        auto sl = streamline();
        advance(sl);
        const auto first = present_frame(owner, ++present);
        if (first.latest != sl.pending || !pending_result(first) || owner.admission_after) return false;
        auto &completed = *sl.pending;
        advance(sl);
        if (!completed_pick(present_frame(owner, ++present), &completed) || owner.provider != provider_kind::streamline) return false;
      }
      // (e) A delivering owner keeps its live source while the other provider
      // is live and readable every frame, including over a pending Present.
      for (const auto owner_kind : {provider_kind::ngx, provider_kind::streamline}) {
        restart();
        auto &owner = fresh_owner();
        auto own = owner_kind == provider_kind::ngx ? ngx() : streamline();
        auto other = owner_kind == provider_kind::ngx ? streamline() : ngx();
        if (!deliver(owner, own, present)) return false;
        for (unsigned frame = 0; frame != 6; ++frame) {
          const bool owner_advances = frame % 3 != 2; // Every third Present has no new completion.
          if (owner_advances) advance(own);
          advance(other); finish_gpu(other);
          const auto chosen = present_frame(owner, ++present);
          if ((chosen.value && chosen.value->metadata.provider != owner_kind) ||
              (owner_advances ? chosen.result != status::ready : !pending_result(chosen)) ||
              owner.provider != owner_kind || owner.admission_after) return false;
        }
        // Ownership is revocable: a failed owner attempt, or no copied owner
        // frame within the source age, lets the readable live source take over.
        auto &failed = advance(own);
        finish(failed.id, false);
        advance(other); finish_gpu(other);
        const auto failover = present_frame(owner, ++present);
        if (!failover.value || failover.value->metadata.provider == owner_kind || failover.result != status::ready) return false;
      }
      {
        restart();
        auto &owner = fresh_owner();
        auto own = ngx(); auto other = streamline();
        if (!deliver(owner, own, present)) return false;
        present_frame(owner, ++present); // Consumes nothing new: the pipeline did not advance.
        owner.last_source_tick -= sunshine_scene_depth::maximum_source_age_ms + 1;
        advance(other); finish_gpu(other);
        const auto takeover = present_frame(owner, ++present);
        if (takeover.value != other.pending || takeover.result != status::ready ||
            owner.provider != provider_kind::streamline) return false;
      }
      // A readable owner whose frames are never copied (failed display
      // preparation or copy, so no complete_frame) is not delivering: once its
      // last copied frame is older than the source age, readable SL takes over.
      {
        restart();
        auto &owner = fresh_owner();
        auto own = ngx(); auto other = streamline();
        if (!deliver(owner, own, present)) return false;
        auto &uncopied = *own.pending;
        advance(own);
        if (!completed_pick(present_frame(owner, ++present, {}, false), &uncopied)) return false;
        owner.last_source_tick -= sunshine_scene_depth::maximum_source_age_ms + 1;
        advance(own); advance(other); finish_gpu(other);
        const auto revoked = present_frame(owner, ++present);
        if (revoked.value != other.pending || revoked.result != status::ready ||
            owner.provider != provider_kind::streamline) return false;
      }
      // An owner's already copied frame is not a pending successor. A one-frame
      // stray owner therefore does not hold a readable live source of the other
      // provider for the source age.
      for (const auto owner_kind : {provider_kind::ngx, provider_kind::streamline}) {
        restart();
        auto &owner = fresh_owner();
        auto stray = owner_kind == provider_kind::ngx ? ngx() : streamline();
        auto other = owner_kind == provider_kind::ngx ? streamline() : ngx();
        advance(stray); finish_gpu(stray);
        if (present_frame(owner, ++present).value != stray.pending || owner.provider != owner_kind) return false;
        for (unsigned frame = 0; frame != 2; ++frame) {
          advance(other); finish_gpu(other);
          const auto chosen = present_frame(owner, ++present);
          if (chosen.value != other.pending || chosen.result != status::ready || owner.provider == owner_kind) return false;
        }
      }
      // The nested fallback in a pipelined game: one owner copy is rejected
      // (a metadata-only nomination, or an attempt rejected before its capture
      // slot exists), so the other provider's inner evaluation is captured and
      // is still in flight at that Present. The delivering owner keeps
      // selection and the display holds its last copied frame. The pixel-less
      // inner capture takes no authority, so the held depth is not invalidated
      // as a source change and selection does not flip to the other provider
      // and back.
      for (const auto owner_kind : {provider_kind::streamline, provider_kind::ngx})
      for (const bool slotless : {false, true}) {
        restart();
        auto &owner = fresh_owner();
        auto own = owner_kind == provider_kind::ngx ? ngx() : streamline();
        auto nested = owner_kind == provider_kind::ngx ? streamline() : ngx();
        if (!deliver(owner, own, present)) return false;
        const auto copied = retained.capture_id;
        // A pick without any capture names no source (epoch zero), so it
        // cannot contradict the retained one.
        const auto owned = [&] {
          return owner.provider_established && owner.provider == owner_kind && owner.source_id == own.source_id &&
            (!decided.epoch || (decided.provider == owner_kind && decided.source_id == own.source_id));
        };
        const auto holds = [&](const capture_pick &chosen) {
          const auto update = display(chosen);
          return update.action == display_action::hold && retained.capture_id == copied;
        };
        slot *rejected{};
        if (slotless) reject_without_slot(own);
        else {
          rejected = &advance(own);
          rejected->nomination_only = true; rejected->pixel_failure = status::unsupported_resource;
        }
        advance(nested);
        const auto held = present_frame(owner, ++present);
        if (held.value != rejected || (slotless && (held.latest || held.pending_nomination)) ||
            held.result != (slotless ? status::unsupported_state : status::unsupported_resource) ||
            !owned() || !holds(held)) return false;
        // The owner resumes; the inner evaluation stops and its capture completes.
        finish_gpu(nested);
        auto &resumed = advance(own);
        const auto pending_owner = present_frame(owner, ++present);
        if (pending_owner.result == status::ready || !owned() || !holds(pending_owner)) return false;
        advance(own);
        const auto fresh = present_frame(owner, ++present);
        if (!completed_pick(fresh, &resumed) || !owned() || display(fresh).action != display_action::copy_fresh ||
            retained.capture_id != resumed.id) return false;
      }
      // A live owner whose attempts fail (rejected nomination-only copies,
      // failed evaluations, a concurrently evaluated second view, or attempts
      // rejected before a capture slot exists) keeps selection only while it
      // is delivering. Once its last copied frame is
      // older than the source age, it holds nothing in front of a pipelined
      // source of the other provider: returning the failing owner would mark a
      // gap at every Present and keep the other source's completed snapshots
      // unreadable. The pending source is reported instead (its submitted
      // nomination may carry source authority, never pixels), so its first
      // completed snapshot after the gap is selected, whether it started while
      // the owner was still delivering or only after it stopped.
      for (const auto owner_kind : {provider_kind::streamline, provider_kind::ngx})
      for (unsigned failure = 0; failure != 4; ++failure)
      for (const bool other_first : {true, false}) {
        restart();
        auto &owner = fresh_owner();
        auto own = owner_kind == provider_kind::ngx ? ngx() : streamline();
        auto other = owner_kind == provider_kind::ngx ? streamline() : ngx();
        auto second_view = owner_kind == provider_kind::ngx ? ngx(11) : streamline(2);
        second_view.first = 16;
        if (!deliver(owner, own, present)) return false;
        const auto fail = [&] {
          if (failure == 3) { reject_without_slot(own); return; }
          auto &value = advance(own);
          if (failure == 0) { value.nomination_only = true; value.pixel_failure = status::unsupported_resource; }
          else if (failure == 1) finish(value.id, false);
          else { second_view.sequence = own.sequence; advance(second_view); own.sequence = second_view.sequence; }
        };
        // While delivering, the failing owner keeps selection and marks the gap.
        // A slot-less owner pick names no capture; the other provider's pick
        // would name its pending one.
        for (unsigned frame = 0; frame != 2; ++frame) {
          fail();
          if (other_first) advance(other);
          const auto kept = present_frame(owner, ++present);
          const auto *source = kept.value ? kept.value : kept.latest;
          if ((source ? source->metadata.provider != owner_kind : failure != 3 || kept.pending_nomination) ||
              kept.result == status::ready || pending_result(kept) ||
              owner.provider != owner_kind || !owner.admission_after) return false;
        }
        // No owner frame was copied within the source age.
        owner.last_source_tick -= sunshine_scene_depth::maximum_source_age_ms + 1;
        const auto gap = owner.admission_after;
        fail(); advance(other);
        const auto handover = present_frame(owner, ++present);
        if ((handover.value && handover.value != other.pending) || handover.latest != other.pending ||
            !pending_result(handover) || owner.admission_after != gap) return false;
        for (unsigned frame = 0; frame != 2; ++frame) {
          auto &completed = *other.pending;
          fail(); advance(other);
          if (!completed_pick(present_frame(owner, ++present), &completed) ||
              owner.provider == owner_kind || owner.admission_after != gap) return false;
        }
      }
      // Required FG keeps its mandatory SL scope: a pipelined FG source uses
      // its own completed snapshot, and NGX is never substituted meanwhile.
      {
        restart();
        auto &owner = fresh_owner();
        auto fg = streamline((1ull << 63) | 0); fg.frame_generation = true;
        auto readable_ngx = ngx();
        const selection_policy policy{true, epoch, 0};
        advance(fg);
        advance(readable_ngx); finish_gpu(readable_ngx);
        const auto waiting = present_frame(owner, ++present, policy);
        if ((waiting.value && !waiting.value->metadata.frame_generation_input) || waiting.latest != fg.pending ||
            !pending_result(waiting)) return false;
        auto &completed = *fg.pending;
        advance(fg);
        if (!completed_pick(present_frame(owner, ++present, policy), &completed) || owner.provider != provider_kind::streamline) return false;
      }
      // (g) Dynamic resolution (Assassin's Creed Shadows with FG on, 10-23): a
      // pipelined SR source on another queue changes its allocation every
      // second real frame (2206x1240, 2442x1374, 2658x1494) and its extent
      // within one, and FG presents a generated frame after each real one.
      // The previous size's completed snapshot is the real frame's fresh depth
      // while the new size is pending, and the generated frame holds that copy:
      // depth on every Present. Refusing either (layout_changed, then the
      // display's depth_shape_changed) dropped depth for 125-438 ms six times
      // in 6 s. Another storage family is another encoding: it is refused until
      // a snapshot of that family completes, and then re-establishes.
      {
        restart();
        auto &owner = fresh_owner();
        auto sr = streamline();
        const source_ref sizes[] = {layout(2206, 1240), layout(2442, 1374), layout(2658, 1494)};
        sr.source = sizes[0];
        if (!deliver(owner, sr, present)) return false;
        for (unsigned frame = 0; frame != 18; ++frame) {
          sr.source = sizes[(frame / 2) % 3];
          auto &completed = *sr.pending;
          auto &next = advance(sr);
          // Every third frame renders a smaller extent of its allocation.
          if (frame % 3 == 2) next.metadata.resource.area = {0, 0, 1792, 1008};
          const auto real = present_frame(owner, ++present);
          if (!completed_pick(real, &completed) || display(real).action != display_action::copy_fresh ||
              retained.capture_id != completed.id) return false;
          const auto generated = present_frame(owner, ++present);
          const auto held = display(generated);
          if (generated.result == status::ready || !generated.value || generated.value != sr.pending ||
              held.action != display_action::hold || std::string_view(held.reason) != "retained_copy" ||
              retained.capture_id != completed.id) return false;
        }
        auto packed = layout(2442, 1374);
        packed->desc.Format = DXGI_FORMAT_R24G8_TYPELESS;
        sr.source = packed;
        advance(sr);
        const auto refused = present_frame(owner, ++present);
        const auto refusal = display(refused);
        if (refused.result == status::ready || refused.selection != selection_reason::format_changed ||
            refusal.action != display_action::invalidate || std::string_view(refusal.reason) != "depth_format_changed")
          return false;
        retained = {}; // The provider's cache invalidates; only a fresh copy re-arms it.
        auto &first_packed = *sr.pending;
        advance(sr);
        const auto recovered = present_frame(owner, ++present);
        if (!completed_pick(recovered, &first_packed) || display(recovered).action != display_action::copy_fresh ||
            retained.capture_id != first_packed.id) return false;
      }
      return true;
    }
    namespace {
      struct private_object final : ID3D12Object {
        ULONG references{1};
        std::uint64_t cookie{};
        unsigned writes{};
        bool refuse_write{};
        HRESULT read_failure{S_OK};
        struct bytes_entry { GUID key{}; std::uint64_t value{}; bool present{}; };
        std::array<bytes_entry,4> private_bytes{};
        GUID interface_key{};
        IUnknown *private_interface{};
        ~private_object() { if (private_interface) private_interface->Release(); }
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
          if (!out) return E_POINTER;
          *out = nullptr;
          if (IsEqualIID(iid, IID_IUnknown) || IsEqualIID(iid, IID_ID3D12Object)) { *out = this; AddRef(); return S_OK; }
          return E_NOINTERFACE;
        }
        ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
        ULONG STDMETHODCALLTYPE Release() override { return --references; }
        HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID key, UINT *size, void *out) override {
          if (FAILED(read_failure)) return read_failure;
          if (!size) return E_POINTER;
          if (private_interface && IsEqualGUID(key, interface_key)) {
            if (!out || *size < sizeof(private_interface)) { *size = sizeof(private_interface); return DXGI_ERROR_MORE_DATA; }
            *size = sizeof(private_interface); private_interface->AddRef();
            *static_cast<IUnknown **>(out) = private_interface; return S_OK;
          }
          const std::uint64_t *value = IsEqualGUID(key, source_guid) && cookie ? &cookie : nullptr;
          for (const auto &entry : private_bytes) if (entry.present && IsEqualGUID(key, entry.key)) value = &entry.value;
          if (!value) return DXGI_ERROR_NOT_FOUND;
          if (!out || *size < sizeof(*value)) { *size = sizeof(*value); return DXGI_ERROR_MORE_DATA; }
          *size = sizeof(*value); *static_cast<std::uint64_t *>(out) = *value; return S_OK;
        }
        HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID key, UINT size, const void *data) override {
          if (refuse_write) return E_FAIL;
          if (size != sizeof(cookie) || !data) return E_INVALIDARG;
          if (IsEqualGUID(key, source_guid)) { cookie = *static_cast<const std::uint64_t *>(data); ++writes; return S_OK; }
          for (auto &entry : private_bytes) if (!entry.present || IsEqualGUID(key, entry.key)) {
            entry = {key, *static_cast<const std::uint64_t *>(data), true}; return S_OK;
          }
          return E_FAIL;
        }
        HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID key, const IUnknown *data) override {
          if (refuse_write || (private_interface && !IsEqualGUID(key, interface_key))) return E_FAIL;
          auto *replacement = const_cast<IUnknown *>(data);
          if (replacement) replacement->AddRef();
          if (private_interface) private_interface->Release();
          private_interface = replacement; interface_key = key;
          return S_OK;
        }
        HRESULT STDMETHODCALLTYPE SetName(LPCWSTR) override { return E_NOTIMPL; }
      };
      std::uint64_t native_object(private_object &value) { return reinterpret_cast<std::uint64_t>(&value); }
      bool retired_recording_regression() {
        auto producer = std::make_shared<sunshine_native_command::recording_lifetime>();
        auto consumer = std::make_shared<sunshine_native_command::recording_lifetime>();
        producer->cookie = 11; consumer->cookie = 22;
        slot submitted;
        submitted.id = 1; submitted.command = 11; submitted.producer_recording = producer;
        submitted.producer_submitted = submitted.acquired = submitted.retirement_unknown = true;
        submitted.producer_fence = 7; submitted.retirements[0] = {77, 13};
        submitted.consumers[0] = 22; submitted.consumer_recordings[0] = consumer;
        retire_dead_recordings(submitted);
        if (submitted.command != 11 || submitted.consumers[0] != 22) return false;
        producer->cookie = consumer->cookie = 0;
        retire_dead_recordings(submitted);
        if (submitted.command || submitted.producer_recording || submitted.consumers[0] || submitted.consumer_recordings[0] ||
            submitted.invalid || !submitted.producer_submitted || !submitted.retirement_unknown ||
            submitted.producer_fence != 7 || submitted.retirements[0].value != 13 || reclaimable(submitted)) return false;
        slot abandoned;
        abandoned.id = 2; abandoned.command = 11; abandoned.producer_recording = producer;
        retire_dead_recordings(abandoned);
        if (abandoned.command || !abandoned.invalid || !abandoned.finished || abandoned.producer_submitted) return false;
        // A discarded CPU consumer can retire its recording reference, but the
        // producer fence is still required and is never advanced by destruction.
        slot consumer_only;
        consumer_only.id = 3; consumer_only.producer_submitted = consumer_only.acquired = true;
        consumer_only.producer_fence = 17;
        consumer_only.consumers[0] = 22; consumer_only.consumer_recordings[0] = consumer;
        retire_dead_recordings(consumer_only);
        return !consumer_only.consumers[0] && !consumer_only.consumer_recordings[0] && !consumer_only.invalid &&
          consumer_only.producer_submitted && consumer_only.producer_fence == 17 &&
          std::none_of(consumer_only.retirements.begin(), consumer_only.retirements.end(), [](const auto &point) { return point.queue != 0; });
      }
      bool command_population_regression() {
        const auto baseline = command_storage::live_count();
        std::array<std::unique_ptr<private_object>,160> objects;
        std::array<recording_ref,160> lives;
        for (unsigned i=0; i!=objects.size(); ++i) {
          objects[i] = std::make_unique<private_object>();
          const auto cookie = ++serial;
          if (!native_observer::set_recording_cookie(native_object(*objects[i]), cookie)) return false;
          auto owner = command(native_object(*objects[i]), cookie, true);
          if (!owner || owner->invalid || owner->observation_generation != command_generation) return false;
          lives[i] = owner.life();
          close(native_object(*objects[i]), cookie, S_OK);
          if (!owner->closed || lives[i]->cookie.load() != cookie) return false;
        }
        if (command_storage::live_count() != baseline + objects.size()) return false;
        for (unsigned i=0; i!=objects.size(); ++i) {
          const auto retired = lives[i];
          const auto old_cookie = retired->cookie.load();
          slot submitted;
          submitted.id = 1; submitted.command = old_cookie; submitted.producer_recording = retired;
          submitted.producer_submitted = true; submitted.producer_fence = 7;
          objects[i].reset(); // Native private-data teardown; no explicit destruction callback.
          if (retired->cookie.load() != 0 || command_storage::live_count() != baseline + objects.size() - 1) return false;
          retire_dead_recordings(submitted);
          if (submitted.command || submitted.invalid || !submitted.producer_submitted || submitted.producer_fence != 7) return false;
          objects[i] = std::make_unique<private_object>();
          const auto cookie = ++serial;
          auto owner = command(native_object(*objects[i]), cookie, true);
          if (!owner || owner->invalid || cookie == old_cookie || retired->cookie.load() != 0) return false;
          lives[i] = owner.life();
        }
        for (auto &object : objects) object.reset();
        return command_storage::live_count() == baseline &&
          std::all_of(lives.begin(), lives.end(), [](const auto &life) { return life->cookie.load() == 0; });
      }
    }
    bool initial_recording_regression() {
      private_object native, resource;
      const auto address = native_object(native);
      if (source_cookie(&resource) || native_observer::get_recording_cookie(address) ||
          !native_observer::recording_cookie_absent(address)) return false;
      // These are the production entry and post-call callbacks for a newly
      // created, already-open list. There has been no Reset or SDK nomination.
      std::uint64_t cookie{};
      associate_recording(address, &cookie);
      auto owner = command(address, cookie, false);
      if (!cookie || !owner || owner->closed || owner->invalid || !owner->states.empty() ||
          native_observer::get_recording_cookie(address) != cookie || owner.life()->cookie.load() != cookie) return false;
      D3D12_RESOURCE_BARRIER transition{};
      transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      transition.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&resource);
      transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      transition.Transition.StateAfter = sampled_state;
      // A list ReShade never reported cannot admit a capture: nothing is tracked.
      barriers(address, cookie, 1, &transition);
      if (source_cookie(&resource) || !owner->states.empty() || owner->native_barrier_calls) return false;
      owner->reported = true; // ReShade's create event.
      barriers(address, cookie, 1, &transition);
      const auto source = source_cookie(&resource);
      const auto *observed = state(*owner, source, false);
      if (!source || resource.writes != 1 || !observed || !observed->known || observed->blocked ||
          observed->value != sampled_state || retain_source_cookie(&resource) != source || resource.writes != 1) return false;
      if (owner->native_barrier_calls != 1 || owner->native_transition_count != 1 || owner->last_barrier_command != address) return false;
      private_object other_command;
      const auto other = command(native_object(other_command), ++serial, true);
      if (!other || state(*other, source, false)) return false; // Another list's barrier is never a state proof.
      // Entry association never invents a missing resource state or overwrites
      // an already associated recording, including malformed zero data.
      std::uint64_t duplicate = 99;
      associate_recording(address, &duplicate);
      if (duplicate || owner->cookie != cookie) return false;
      close(address, cookie, S_OK);
      if (!owner->closed) return false;
      reset(address, cookie, S_OK, true);
      const auto next_cookie = native_observer::get_recording_cookie(address);
      if (!next_cookie || next_cookie == cookie || owner->closed || !owner->states.empty() ||
          owner->native_barrier_calls || owner->native_transition_count || owner->last_barrier_command) return false;
      // A post-call callback carrying the entry's old or zero identity cannot
      // attach to the new recording after Reset. This applies to state, close,
      // render-pass and enhanced-barrier evidence alike.
      const native_observer::enhanced_texture enhanced_texture{reinterpret_cast<ID3D12Resource *>(&resource),
        D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE, true};
      for (const auto stale : {std::uint64_t{}, cookie}) {
        barriers(address, stale, 1, &transition);
        close(address, stale, S_OK);
        render_pass(address, stale, true);
        enhanced_textures(address, stale, 1, &enhanced_texture);
      }
      if (owner->closed || owner->render_pass || owner->invalid || !owner->states.empty()) return false;
      render_pass(address, next_cookie, true);
      if (!owner->render_pass) return false;
      render_pass(address, next_cookie, false);
      // An enhanced texture barrier sets the texture's exact layout and access
      // (value is only the legacy equivalent, for diagnostics).
      const native_observer::enhanced_texture shader_resource{reinterpret_cast<ID3D12Resource *>(&resource),
        D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, true, D3D12_BARRIER_ACCESS_SHADER_RESOURCE};
      enhanced_textures(address, next_cookie, 1, &shader_resource);
      const auto *mapped = state(*owner, source_cookie(&resource), false);
      if (owner->render_pass || owner->invalid || !mapped || mapped->blocked || !mapped->known || !mapped->enhanced ||
          mapped->layout != D3D12_BARRIER_LAYOUT_SHADER_RESOURCE || mapped->access != D3D12_BARRIER_ACCESS_SHADER_RESOURCE ||
          mapped->value != (D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE))
        return false;
      const native_observer::enhanced_texture stencil_only{reinterpret_cast<ID3D12Resource *>(&resource),
        D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE, false};
      enhanced_textures(address, next_cookie, 1, &stencil_only);
      if (mapped->blocked || !mapped->known || mapped->layout != D3D12_BARRIER_LAYOUT_SHADER_RESOURCE) return false;
      // A direct-queue layout is the capture's own queue type: admitted as is.
      enhanced_textures(address, next_cookie, 1, &enhanced_texture);
      const auto *enhanced = state(*owner, source_cookie(&resource), false);
      if (owner->render_pass || owner->invalid || !enhanced || enhanced->blocked || !enhanced->known ||
          enhanced->layout != D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE) return false;
      // Layouts the copy cannot use refuse it until a later barrier names an
      // admissible one; they never block the texture for the recording.
      for (const auto layout : {D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE, D3D12_BARRIER_LAYOUT_UNDEFINED,
          D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ, D3D12_BARRIER_LAYOUT_RESOLVE_DEST, D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON}) {
        const native_observer::enhanced_texture refused_layout{reinterpret_cast<ID3D12Resource *>(&resource), layout, true};
        enhanced_textures(address, next_cookie, 1, &refused_layout);
        if (owner->invalid || enhanced->blocked || enhanced->known || enhanced->layout != std::uint32_t(layout)) return false;
        enhanced_textures(address, next_cookie, 1, &shader_resource);
        if (enhanced->blocked || !enhanced->known) return false;
      }
      const native_observer::enhanced_texture no_access{reinterpret_cast<ID3D12Resource *>(&resource),
        D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, true, D3D12_BARRIER_ACCESS_NO_ACCESS};
      enhanced_textures(address, next_cookie, 1, &no_access);
      if (enhanced->blocked || enhanced->known) return false;
      // A split (SyncAfter SPLIT) is in transition until its end barrier.
      native_observer::enhanced_texture begin_split{reinterpret_cast<ID3D12Resource *>(&resource),
        D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE, true, D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
        D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_SPLIT};
      enhanced_textures(address, next_cookie, 1, &begin_split);
      if (enhanced->blocked || enhanced->known || !enhanced->pending_split) return false;
      auto end_split = begin_split; end_split.sync_before = D3D12_BARRIER_SYNC_SPLIT; end_split.sync_after = D3D12_BARRIER_SYNC_ALL;
      enhanced_textures(address, next_cookie, 1, &end_split);
      if (enhanced->blocked || !enhanced->known || enhanced->pending_split) return false;
      // A legacy barrier inside an enhanced split is outside the contract: blocked until Reset.
      enhanced_textures(address, next_cookie, 1, &begin_split);
      barriers(address, next_cookie, 1, &transition);
      enhanced_textures(address, next_cookie, 1, &shader_resource);
      if (!enhanced->blocked || enhanced->known) return false;

      private_object closed, pass, malformed, refused, orphan;
      std::uint64_t first{};
      associate_recording(native_object(closed), &first); close(native_object(closed), first, S_OK);
      auto closed_owner = command(native_object(closed), first, false);
      if (!first || !closed_owner || !closed_owner->closed) return false;
      associate_recording(native_object(pass), &first); render_pass(native_object(pass), first, true);
      auto pass_owner = command(native_object(pass), first, false);
      if (!first || !pass_owner || !pass_owner->render_pass) return false;
      if (!native_observer::set_recording_cookie(native_object(malformed), 0) ||
          native_observer::recording_cookie_absent(native_object(malformed))) return false;
      associate_recording(native_object(malformed), &first);
      if (first || command_storage::acquire(&malformed, false)) return false;
      refused.refuse_write = true;
      associate_recording(native_object(refused), &first);
      if (first || native_observer::get_recording_cookie(native_object(refused))) return false;
      auto orphan_owner = command(native_object(orphan), ++serial, true);
      associate_recording(native_object(orphan), &first);
      return orphan_owner && !first && !native_observer::get_recording_cookie(native_object(orphan));
    }
    bool recording_recovery_regression() {
      // Exercise actual global loss and allocation together. Existing named
      // recordings remain rejected; a distinct new recording starts clean.
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        status saved_reason{reason.load()};
        std::uint64_t saved_generation{command_generation};
        ~cleanup() { slots = saved_slots; reason = saved_reason; command_generation = saved_generation; }
      } restore;
      private_object old_object, fresh_object;
      const auto old_cookie = ++serial;
      auto old = command(native_object(old_object), old_cookie, true);
      if (!old) return false;
      old->closed = old->render_pass = true;
      old->states.push_back({++serial, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true});
      invalidate_all();
      auto fresh = command(native_object(fresh_object), ++serial, true);
      if (!fresh || &*fresh == &*old || fresh->invalid || fresh->closed || fresh->render_pass ||
          std::any_of(fresh->states.begin(), fresh->states.end(), [](const auto &v) { return v.source || v.value || v.known || v.blocked; })) return false;
      auto lost = command(native_object(old_object), old_cookie, true);
      if (!lost || &*lost != &*old || !old->invalid || !old->closed || !old->render_pass || !old->states[0].known) return false;
      reset(native_object(old_object), old_cookie, S_OK, false);
      const auto new_cookie = native_observer::get_recording_cookie(native_object(old_object));
      auto reused = command(native_object(old_object), new_cookie, true);
      return new_cookie && new_cookie != old_cookie && reused && &*reused == &*old &&
        !command(native_object(old_object), old_cookie, false) && !reused->invalid && !reused->closed && !reused->render_pass &&
        reused.life()->cookie.load() == new_cookie &&
        std::none_of(reused->states.begin(), reused->states.end(), [](const auto &v) { return v.source || v.value || v.known || v.blocked; }) &&
        command_population_regression() && retired_recording_regression();
    }
    bool recording_state_growth_regression() {
      private_object command_object, target;
      std::array<private_object, 160> unrelated;
      const auto native_command = native_object(command_object);
      const auto cookie = ++serial;
      auto recording = command(native_command, cookie, true);
      if (!recording) return false;
      recording->reported = true; // ReShade's create event.
      const auto target_id = retain_source_cookie(&target);
      D3D12_RESOURCE_BARRIER transition{};
      transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      transition.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&target);
      transition.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
      barriers(native_command, cookie, 1, &transition);
      const auto append_unrelated = [&](unsigned first, unsigned end) {
        for (unsigned i = first; i != end; ++i) {
          D3D12_RESOURCE_BARRIER split = transition;
          split.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
          split.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&unrelated[i]);
          barriers(native_command, cookie, 1, &split);
          const auto *stored = state(*recording, unrelated[i].cookie, false);
          if (recording->invalid || !stored || !stored->pending_split || stored->known || stored->blocked) return false;
        }
        return true;
      };
      if (!append_unrelated(0, 80) || recording->states.size() != 81) return false;
      // Unrelated barriers beyond the old cap must leave the target's capture
      // admission facts intact: open, valid recording and known nonzero state.
      const auto *selected = state(*recording, target_id, false);
      if (recording->invalid || recording->closed || recording->render_pass || !selected ||
          !selected->known || selected->blocked || selected->value != transition.Transition.StateAfter) return false;
      // A split barrier leaves the target in transition across further storage
      // growth, until a later ordinary transition defines its state.
      auto split = transition;
      split.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
      barriers(native_command, cookie, 1, &split);
      if (!append_unrelated(80, unsigned(unrelated.size()))) return false;
      selected = state(*recording, target_id, false);
      if (recording->invalid || !selected || !selected->pending_split || selected->known) return false;
      barriers(native_command, cookie, 1, &transition);
      selected = state(*recording, target_id, false);
      if (recording->invalid || !selected || selected->pending_split || !selected->known || selected->blocked ||
          recording->states.size() != 161) return false;
      const auto capacity = recording->states.capacity();
      const auto *storage = recording->states.data();
      reset(native_command, cookie, S_OK, true);
      const auto next_cookie = native_observer::get_recording_cookie(native_command);
      if (!next_cookie || next_cookie == cookie || recording->cookie != next_cookie || recording->invalid ||
          !recording->states.empty() || recording->states.capacity() != capacity || recording->states.data() != storage) return false;
      barriers(native_command, next_cookie, 1, &transition);
      selected = state(*recording, target_id, false);
      return selected && selected->known && !selected->blocked && !recording->invalid &&
        recording->states.capacity() == capacity && recording->states.data() == storage;
    }
    // The split matrix, through the production barrier callbacks and the
    // capture's state table: a split in transition is refused, its END_ONLY
    // or a later plain transition defines the state, a split of another
    // subresource leaves subresource 0, and nothing clears a block.
    bool split_barrier_regression() {
      private_object command_object;
      const auto native_command = native_object(command_object);
      const auto cookie = ++serial;
      auto recording = command(native_command, cookie, true);
      if (!recording) return false;
      constexpr auto read = D3D12_RESOURCE_STATE_DEPTH_READ;
      const auto selected = copy_state::select(sunshine_scene_depth::state_proof::observed_nonzero,
        texture_state_policy::source_contract);
      const auto run = [&](std::initializer_list<D3D12_RESOURCE_BARRIER> sequence, bool enhanced_block = false) {
        recording->restart(cookie, command_generation);
        recording->reported = true;
        recording.life()->cookie.store(cookie);
        private_object depth;
        if (enhanced_block) {
          // A legacy barrier inside an enhanced split blocks until Reset.
          const native_observer::enhanced_texture split{reinterpret_cast<ID3D12Resource *>(&depth),
            D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ, true, D3D12_BARRIER_ACCESS_DEPTH_STENCIL_READ,
            D3D12_BARRIER_SYNC_ALL, D3D12_BARRIER_SYNC_SPLIT};
          enhanced_textures(native_command, cookie, 1, &split);
        }
        for (auto value : sequence) {
          value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&depth);
          barriers(native_command, cookie, 1, &value);
        }
        if (recording->invalid) return copy_state::reject(status::failed, record_stage::recording_invalid);
        return copy_state::resolve(selected, 0, observation_of(state(*recording, depth.cookie, false)),
          D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
      };
      const auto barrier = [](D3D12_RESOURCE_BARRIER_FLAGS flags, D3D12_RESOURCE_STATES after,
          UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES) {
        D3D12_RESOURCE_BARRIER value{};
        value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; value.Flags = flags;
        value.Transition.Subresource = subresource;
        value.Transition.StateBefore = D3D12_RESOURCE_STATE_DEPTH_WRITE; value.Transition.StateAfter = after;
        return value;
      };
      constexpr auto begin = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY, end = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
      constexpr auto none = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      const auto admitted = [](const copy_state::decision &d) { return d.admitted && d.state == read && d.used_observed; };
      const auto in_transition = [](const copy_state::decision &d) {
        return !d.admitted && d.result == status::incomplete_state && d.stage == record_stage::incomplete_state;
      };
      return in_transition(run({barrier(begin, read)})) &&
        admitted(run({barrier(begin, read), barrier(end, read)})) &&
        admitted(run({barrier(begin, D3D12_RESOURCE_STATE_COPY_SOURCE), barrier(none, read)})) &&
        admitted(run({barrier(end, read)})) &&
        admitted(run({barrier(none, read), barrier(begin, D3D12_RESOURCE_STATE_COPY_SOURCE, 1)})) &&
        admitted(run({barrier(none, read, 0), barrier(begin, D3D12_RESOURCE_STATE_COPY_SOURCE, 1), barrier(end, D3D12_RESOURCE_STATE_COPY_SOURCE, 1)})) &&
        in_transition(run({barrier(begin, read), barrier(end, read)}, true)) &&
        in_transition(run({barrier(begin | end, read), barrier(end, read)}));
    }
    bool recording_state_loss_regression() {
      // Call the production observer callback before any source cookie exists.
      // Private-data storage is a COM stub; no native API/GPU work is issued.
      private_object native, unrelated, alias_peer, refused, command_object;
      const auto native_command = native_object(command_object);
      const auto cookie = ++serial;
      auto recording = command(native_command, cookie, true);
      auto capture = std::find_if(slots.begin(), slots.end(), [](const auto &v) { return !v.id; });
      if (!recording || capture == slots.end()) return false;
      struct cleanup {
        slot *capture;
        ~cleanup() { *capture = {}; }
      } restore{&*capture};
      const auto restart_case = [&] {
        recording->restart(cookie, command_generation);
        recording->reported = true; // ReShade's lifecycle drives this recording.
        recording.life()->cookie.store(cookie);
      };
      recording->reported = true;
      *capture = {}; // Match record() allocation after a prior global invalidation.
      capture->id = ++serial; capture->command = cookie; capture->finished = capture->success = capture->producer_submitted = true;
      D3D12_RESOURCE_BARRIER value{};
      value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      // All calls made here use ID3D12Object's inherited private-data methods.
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      value.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barriers(native_command, cookie, 1, &value);
      const auto *ordinary = state(*recording, native.cookie, false);
      if (recording->invalid || !native.cookie || !ordinary || !ordinary->known || ordinary->blocked ||
          ordinary->value != D3D12_RESOURCE_STATE_UNORDERED_ACCESS || capture->invalid) return false;
      // Pre-nomination identity is not permission to invent another plane's
      // state, or to reinterpret unsupported native states as ordinary ones.
      private_object different_plane;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&different_plane);
      value.Transition.Subresource = 1;
      barriers(native_command, cookie, 1, &value);
      if (different_plane.cookie || recording->invalid) return false;
      value.Transition.Subresource = 0;
      value.Transition.StateAfter = static_cast<D3D12_RESOURCE_STATES>(0x80000000u);
      barriers(native_command, cookie, 1, &value);
      const auto *unsupported = state(*recording, different_plane.cookie, false);
      if (!different_plane.cookie || !unsupported || unsupported->known || recording->invalid) return false;
      value.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      for (const auto flags : {D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY, D3D12_RESOURCE_BARRIER_FLAG_END_ONLY}) {
        restart_case();
        private_object pretag;
        value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&pretag);
        value.Flags = flags;
        barriers(native_command, cookie, 1, &value);
        const auto *entry = state(*recording, pretag.cookie, false);
        const bool begun = flags == D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
        if (recording->invalid || !pretag.cookie || !entry || entry->blocked || entry->pending_split != begun ||
            entry->known == begun || capture->invalid) return false;
        value.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barriers(native_command, cookie, 1, &value);
        if (entry->blocked || entry->pending_split || !entry->known || recording->invalid || capture->invalid) return false;
      }
      // A flag combination no transition has blocks until Reset, across a
      // later ordinary transition.
      restart_case();
      private_object both;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&both);
      value.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY | D3D12_RESOURCE_BARRIER_FLAG_END_ONLY;
      barriers(native_command, cookie, 1, &value);
      value.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      barriers(native_command, cookie, 1, &value);
      const auto *both_state = state(*recording, both.cookie, false);
      if (recording->invalid || !both_state || !both_state->blocked || capture->invalid) return false;
      // Unrelated named exceptional barriers must leave selected depth usable.
      restart_case();
      const auto selected = retain_source_cookie(&native);
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      barriers(native_command, cookie, 1, &value);
      auto *selected_state = state(*recording, selected, false);
      if (!selected_state || !selected_state->known || selected_state->blocked) return false;
      value.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&unrelated);
      barriers(native_command, cookie, 1, &value);
      // An alias changes no resource state: it neither registers nor blocks
      // its resources, and the unrelated split stays in transition.
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
      value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&unrelated);
      value.Aliasing.pResourceAfter = reinterpret_cast<ID3D12Resource *>(&alias_peer);
      barriers(native_command, cookie, 1, &value);
      const auto *unrelated_state = state(*recording, unrelated.cookie, false);
      selected_state = state(*recording, selected, false);
      if (recording->invalid || selected_state->blocked || !selected_state->known || !unrelated_state ||
          !unrelated_state->pending_split || alias_peer.cookie || capture->invalid) return false;
      // Even a named alias of selected depth keeps its observed state: the SDK
      // consumes the source where it is copied, so the game keeps it active there.
      value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&native);
      barriers(native_command, cookie, 1, &value);
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      value.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      barriers(native_command, cookie, 1, &value);
      selected_state = state(*recording, selected, false);
      if (recording->invalid || !selected_state || selected_state->blocked || !selected_state->known ||
          selected_state->value != D3D12_RESOURCE_STATE_UNORDERED_ACCESS || capture->invalid) return false;
      // Wildcard aliases (NULL before, after or both) likewise leave the recording intact.
      for (unsigned wildcard = 0; wildcard != 3; ++wildcard) {
        restart_case();
        value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
        if (wildcard == 1) value.Aliasing.pResourceBefore = reinterpret_cast<ID3D12Resource *>(&native);
        if (wildcard == 2) value.Aliasing.pResourceAfter = reinterpret_cast<ID3D12Resource *>(&native);
        barriers(native_command, cookie, 1, &value);
        if (recording->invalid || capture->invalid) return false;
      }
      restart_case();
      refused.refuse_write = true;
      value = {}; value.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&refused);
      barriers(native_command, cookie, 1, &value);
      if (!recording->invalid || recording->invalidation != recording_loss::source_identity_unavailable || capture->invalid) return false;
      // Only exact private-data absence permits first-use association.
      private_object malformed, failed_read;
      malformed.private_bytes[0] = {source_guid, 0, true};
      failed_read.read_failure = E_FAIL;
      for (auto *unavailable : {&malformed, &failed_read}) {
        restart_case();
        value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(unavailable);
        barriers(native_command, cookie, 1, &value);
        if (!recording->invalid || recording->invalidation != recording_loss::source_identity_unavailable ||
            unavailable->writes || capture->invalid) return false;
      }
      restart_case();
      value.Flags = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&refused);
      barriers(native_command, cookie, 1, &value);
      if (!recording->invalid || recording->invalidation != recording_loss::source_identity_unavailable || capture->invalid) return false;
      restart_case();
      // Exercise the real allocation-failure boundary without exhausting the
      // test process. Growth success is covered below; failure still closes.
      while (recording->states.size() != recording->states.capacity())
        recording->states.push_back({++serial, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, true});
      struct restore_growth { ~restore_growth() { fail_state_growth = false; } } restore_failure;
      fail_state_growth = true;
      if (!recording->states.empty() && !state(*recording, recording->states.front().source, true)) return false;
      value.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&native);
      barriers(native_command, cookie, 1, &value);
      if (!recording->invalid || recording->invalidation != recording_loss::source_state_capacity || capture->invalid) return false;
      fail_state_growth = false;
      reset(native_command, cookie, S_OK, true);
      const auto new_cookie = native_observer::get_recording_cookie(native_command);
      return new_cookie && new_cookie != cookie && recording->cookie == new_cookie &&
        recording.life()->cookie.load() == new_cookie && !recording->invalid && !capture->invalid && capture->command == 0 &&
        std::none_of(recording->states.begin(), recording->states.end(), [](const auto &v) { return v.blocked || v.source; }) &&
        recording_state_growth_regression() && split_barrier_regression();
    }
    bool source_cookie_reentry_regression() {
      // Real production cookie allocation and command-state lookup, with only
      // ID3D12Object private-data storage mocked. No GPU or source state is faked
      // through the public capture path.
      private_object native, other, refused;
      auto first = std::make_shared<source_reference>();
      first->cookie = retain_source_cookie(&native);
      const auto identity = first->cookie;
      if (!identity || native.writes != 1 ||
          sunshine_native_identity::resource_cookie(&native, true) != identity || native.writes != 1) return false;
      command_state recording;
      auto *before = state(recording, identity, true);
      if (!before) return false;
      before->known = true; before->value = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      std::weak_ptr<source_reference> weak = first;
      first.reset(); // Metadata expires; the native resource remains alive.
      if (!weak.expired()) return false;
      auto returned = std::make_shared<source_reference>();
      returned->cookie = retain_source_cookie(&native);
      const auto *after = state(recording, returned->cookie, false);
      if (returned->cookie != identity || native.writes != 1 || after != before || !after->known ||
          after->value != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) return false;
      const auto fallback = sunshine_native_identity::allocate_identity();
      const auto replacement = retain_source_cookie(&other);
      const auto backup = sunshine_native_identity::allocate_identity();
      refused.refuse_write = true;
      return fallback > identity && replacement > fallback && backup > replacement &&
        sunshine_native_identity::resource_cookie(&native, true) == identity && native.writes == 1 &&
        !state(recording, replacement, false) &&
        retain_source_cookie(&refused) == 0 && refused.cookie == 0;
    }
    bool unsupported_com_boundary_regression() {
      // Only the three IUnknown slots exist. Reaching GetPrivateData/GetDevice
      // or a graphics-list slot without successful QI is a real invalid access.
      struct unknown_only final : IUnknown {
        ULONG references = 1; unsigned queries = 0;
        HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void **out) override {
          ++queries;
          if (!out) return E_POINTER;
          *out = nullptr;
          if (IsEqualIID(iid, IID_IUnknown)) { *out = this; AddRef(); return S_OK; }
          return E_NOINTERFACE;
        }
        ULONG STDMETHODCALLTYPE AddRef() override { return ++references; }
        ULONG STDMETHODCALLTYPE Release() override { return --references; }
      } object;
      const auto native = reinterpret_cast<std::uint64_t>(&object);
      initialize(true);
      observe_command(native); observe_queue(native); command_destroyed(native);
      input value;
      value.epoch = value.sequence = 1;
      value.source = std::make_shared<source_reference>();
      value.projection = {0.0, .0625, true, true};
      value.resource.native = native;
      value.proof = sunshine_scene_depth::state_proof::declared;
      value.native_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
      value.valid_until = sunshine_scene_depth::lifetime::until_present;
      record_diagnostic diagnostic;
      const bool rejected_record = record(native, value, &diagnostic) == 0 && last_status() == status::unsupported_queue &&
        diagnostic.result == status::unsupported_queue && diagnostic.stage == record_stage::command_interface && diagnostic.command == native;
      packet consumer; consumer.ownership = std::make_shared<texture_reference>();
      consumer_diagnostic consumer_diagnostics;
      const auto prior_queries = object.queries;
      const bool rejected_missing_pixels = !mark_consumer(native, consumer, &consumer_diagnostics) &&
        consumer_diagnostics.result == consumer_status::capture_not_ready && object.queries == prior_queries;
      consumer.pixel_ready = true; // Exercise exact QI rejection, not the earlier readiness gate.
      const bool rejected_consumer = !mark_consumer(native, consumer);
      const bool rejected_owner = !provider_active(native);
      SetLastError(731);
      const bool rejected_identity = !sunshine_native_identity::identify_resource(native) && GetLastError() == 731;
      shutdown();
      return rejected_record && rejected_missing_pixels && rejected_consumer && rejected_owner && rejected_identity &&
        object.queries == 7 && object.references == 1;
    }
    bool zero_cookie_submission_regression() {
      std::array<slot, slot_limit> captures;
      captures[0].id = 1; captures[0].producer_submitted = true; // Reset producer; still owned/in flight.
      captures[0].consumers[0] = 44;
      captures[1].id = 2; captures[1].command = 55;
      std::array<bool, slot_limit> producers{}, consumers{};
      const native_observer::command_identity unknown[]{{111, 0}, {222, 0}};
      if (identify_submissions(captures, 2, unknown, producers, consumers) || captures[0].invalid ||
          std::any_of(producers.begin(), producers.end(), [](bool v) { return v; }) ||
          std::any_of(consumers.begin(), consumers.end(), [](bool v) { return v; })) return false;
      const native_observer::command_identity mixed[]{{111, 0}, {333, 55}, {444, 44}};
      if (!identify_submissions(captures, 3, mixed, producers, consumers) || captures[0].invalid ||
          producers[0] || !producers[1] || !consumers[0] || consumers[1]) return false;
      // A real repeated recording must still be rejected as a stale replay.
      captures[1].producer_submitted = true;
      identify_submissions(captures, 3, mixed, producers, consumers);
      return captures[1].invalid && !captures[0].invalid;
    }
    bool cross_queue_completion_regression() {
      // Exercise the same admission and retirement functions used by live
      // acquire/submitted/reclaimable, supplying only native fence results.
      constexpr std::uint64_t producer = 71, consumer = 72, other = 73, device = 41;
      slot value;
      value.id = 1; value.queue = producer; value.producer_fence = 500;
      value.finished = value.success = value.producer_submitted = true;
      const auto admission = [&](std::uint64_t native, std::uint64_t identity, queue_progress progress) {
        return submission_status(value, native, identity, progress);
      };
      if (admission(producer, device, {}) != status::ready ||
          admission(producer, device, {device, UINT64_MAX, true}) != status::ready ||
          admission(consumer, device, {device, 499, true}) != status::submitted ||
          admission(consumer, device, {device, 500, true}) != status::ready ||
          admission(consumer, device, {device, 900, true}) != status::ready ||
          admission(consumer, device + 1, {device, 900, true}) != status::unsupported_queue ||
          admission(consumer, device, {device, UINT64_MAX, true}) != status::failed ||
          admission(consumer, device, {device, 900, false}) != status::failed) return false;
      value.command = 99;
      value.producer_recording = std::make_shared<sunshine_native_command::recording_lifetime>();
      value.producer_recording->cookie = 99;
      if (admission(consumer, device, {device, 900, true}) != status::submitted ||
          admission(producer, device, {}) != status::ready) return false;
      value.producer_recording->cookie = 0; // Destruction retires the exact old recording.
      if (admission(consumer, device, {device, 900, true}) != status::ready) return false;
      value.producer_recording->cookie = 100; // Reset starts a different recording.
      if (admission(consumer, device, {device, 900, true}) != status::ready) return false;
      value.producer_recording.reset();
      if (admission(consumer, device, {device, 900, true}) != status::submitted) return false;
      value.command = 0;
      if (admission(consumer, device, {device, 900, true}) != status::ready) return false;
      value.finished = false;
      if (admission(consumer, device, {device, 900, true}) == status::ready) return false;
      value.finished = true; value.success = false;
      if (admission(consumer, device, {device, 900, true}) != status::failed) return false;
      value.success = true; value.consumer_queue = consumer;
      if (admission(other, device, {device, 900, true}) != status::unsupported_queue ||
          admission(producer, device, {}) != status::unsupported_queue) return false;

      value.command = 101; value.consumers[0] = 102; value.acquired = true;
      consumer_submitted(value, consumer, 7, true);
      consumer_submitted(value, consumer, 11, true); // Coalesce only the same timeline.
      std::uint64_t producer_complete = 500, consumer_complete = 10, other_complete = 2;
      const auto complete = [&](const fence_point &point) {
        const auto done = point.queue == producer ? producer_complete : point.queue == consumer ? consumer_complete :
          point.queue == other ? other_complete : UINT64_MAX;
        return done != UINT64_MAX && done >= point.value;
      };
      if (value.invalid || value.retirements[0].queue != consumer || value.retirements[0].value != 11 ||
          !uses_queue(value, producer) || !uses_queue(value, consumer) || uses_queue(value, other) || gpu_retired(value, complete)) return false;
      retire_recording(value, 101); retire_recording(value, 102);
      if (value.command || value.consumers[0] || value.producer_fence != 500 || value.retirements[0].value != 11 ||
          !uses_queue(value, consumer) || gpu_retired(value, complete)) return false;
      consumer_complete = 11;
      if (!gpu_retired(value, complete)) return false;
      producer_complete = 499;
      if (gpu_retired(value, complete)) return false;
      producer_complete = UINT64_MAX;
      if (gpu_retired(value, complete)) return false;
      producer_complete = 500;
      // An externally replayed consumer on a different queue is invalid, but
      // its actual fence must still pin the copy and that queue until complete.
      consumer_submitted(value, other, 3, true);
      if (!value.invalid || value.failure != capture_failure::consumer_queue_changed ||
          !uses_queue(value, other) || gpu_retired(value, complete)) return false;
      other_complete = 3;
      if (!gpu_retired(value, complete)) return false;
      value.retirement_unknown = true;
      if (gpu_retired(value, complete)) return false;
      value.retirement_unknown = false;
      consumer_submitted(value, consumer, 12, false);
      if (!value.retirement_unknown || gpu_retired(value, complete)) return false;

      slot same_queue;
      same_queue.id = 2; same_queue.queue = producer; same_queue.producer_fence = 500;
      same_queue.consumer_queue = producer;
      consumer_submitted(same_queue, producer, 501, true);
      if (same_queue.invalid || gpu_retired(same_queue, complete)) return false;
      producer_complete = 501;
      if (!gpu_retired(same_queue, complete)) return false;
      // Discarding a consumer before Execute adds no phantom GPU obligation;
      // it still cannot release an unfinished producer.
      slot discarded;
      discarded.id = 3; discarded.queue = producer; discarded.producer_fence = 502;
      discarded.producer_submitted = true; discarded.consumers[0] = 103;
      retire_recording(discarded, 103);
      if (discarded.consumers[0] || gpu_retired(discarded, complete)) return false;
      producer_complete = 502;
      return gpu_retired(discarded, complete);
    }
    bool source_authority_regression() {
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        decltype(evaluations) saved_evaluations{evaluations};
        std::array<status, provider_count> attempts{attempt_status[0].load(), attempt_status[1].load()};
        bool enabled{requested.load()}; status saved_reason{reason.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations; requested = enabled; reason = saved_reason;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = attempts[i];
        }
      } restore;
      fixture::clear_captures(); requested = true;
      auto source = std::make_shared<source_reference>();
      source->cookie = 31; source->device_identity = 41;
      source->desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
      source->desc.Width = 800; source->desc.Height = 600;
      source->desc.DepthOrArraySize = source->desc.MipLevels = source->desc.SampleDesc.Count = 1;
      source->desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // Valid identity/extent, unsupported depth copy.
      input metadata; metadata.source = source; metadata.epoch = 1; metadata.sequence = 1;
      metadata.source_id = 19; metadata.tick = GetTickCount64(); metadata.resource.native = 23;
      copy_region region;
      if (source_region(source->desc, metadata.resource, region) != status::ready ||
          capture_region(source->desc, metadata.resource, region) != status::unsupported_resource) return false;
      auto malformed = metadata.resource; malformed.area = {799, 0, 2, 600};
      if (source_region(source->desc, malformed, region) != status::malformed) return false;
      begin_evaluation(1, 1, 0, provider_kind::streamline, 19);
      const auto ticket = record_nomination(metadata, metadata.resource.native, 77, {});
      const auto nominated = std::find_if(slots.begin(), slots.end(), [&](const auto &value) { return value.id == ticket; });
      if (!ticket || nominated == slots.end()) return false;
      auto &value = *nominated;
      value.source_nominated = value.nomination_only = true; value.pixel_failure = status::unsupported_resource;
      queue_state owner; owner.device_identity = 41;
      constexpr std::uint64_t consumer = 101;
      const auto select = [&] { return select_provider(owner, consumer, 1, GetTickCount64(), {}); };
      if (select().value) return false;
      finish(ticket, true);
      if (select().value) return false; // Successful but never submitted is not source authority.
      producer_submitted(value, consumer + 1, 0, true);
      auto chosen = select();
      if (chosen.value != &value || chosen.result != status::unsupported_resource ||
          nomination_status(value, owner.device_identity + 1) != status::unsupported_queue) return false;
      packet out; describe_packet(value, consumer, out);
      if (out.pixel_ready || out.texture || out.ownership || out.resource_id != 31 ||
          out.width != 800 || out.height != 600) return false;
      consumer_diagnostic consumer_result;
      if (copy_current(5, out, 7, D3D12_RESOURCE_STATE_COPY_DEST, &consumer_result) ||
          consumer_result.result != consumer_status::capture_not_ready) return false;
      establish_provider(owner, value, 1);
      if (!owner.provider_established || owner.source_id != 19) return false;
      // Native pixel submission readiness is independent too: no private fence
      // means no usable snapshot, but the successful submitted source remains.
      value.nomination_only = false; value.queue = consumer;
      if (nomination_status(value, owner.device_identity) != status::ready ||
          submission_status(value, consumer, owner.device_identity, {}) == status::ready) return false;
      value.producer_fence = 7;
      bind_pixel_consumer(value, consumer, submission_status(value, consumer, owner.device_identity, {}));
      if (value.consumer_queue != consumer) return false;
      consumer_submitted(value, consumer, 8, true);
      queue_state other_owner; other_owner.device_identity = owner.device_identity;
      // Q2 can see the same authoritative source, but Q1 already owns the
      // snapshot's transition timeline. Repeating Q2 effects cannot turn that
      // authority-only result into permission to read the same texture.
      for (unsigned attempt = 0; attempt != 2; ++attempt) {
        const auto other = select_provider(other_owner, consumer + 1, 1, GetTickCount64(), {});
        if (other.value != &value || other.result != status::unsupported_queue) return false;
        bind_pixel_consumer(value, consumer + 1, other.result);
        establish_provider(other_owner, value, 1);
        if (value.consumer_queue != consumer ||
            submission_status(value, consumer, owner.device_identity, {}) != status::ready) return false;
      }
      value.nomination_only = true; value.success = false;
      owner.provider_established = false; owner.present_capture = 0;
      if (select().value) return false; // Failed first evaluation still permits fallback.
      value.success = true;
      invalidate(value, capture_failure::producer_signal_failed);
      if (nomination_status(value, owner.device_identity) != status::ready) return false;
      // A later source release must revoke authority even when its first pixel
      // failure remains the diagnostic reason (first-failure preservation).
      invalidate(value, capture_failure::source_retired);
      return nomination_status(value, owner.device_identity) == status::failed;
    }
    bool preservation_owner_regression() {
      struct cleanup {
        std::array<slot, slot_limit> saved_slots{slots};
        decltype(evaluations) saved_evaluations{evaluations};
        std::array<status, provider_count> attempts{attempt_status[0].load(), attempt_status[1].load()};
        status saved_reason{reason.load()};
        ~cleanup() {
          slots = saved_slots; evaluations = saved_evaluations; reason = saved_reason;
          for (unsigned i = 0; i != provider_count; ++i) attempt_status[i] = attempts[i];
        }
      } restore;
      fixture::clear_captures();
      constexpr std::uint64_t device = 11, source_id = 12, producer = 17, consumer = 19, recording = 21;
      auto source = std::make_shared<source_reference>(); source->cookie = source_id; source->device_identity = device;
      auto &value = slots[0];
      value.id = 101; value.preservation_only = true; value.finished = value.success = true;
      value.command = recording; value.metadata.source = source;
      value.metadata.epoch = 3; value.metadata.sequence = 4; value.metadata.source_id = 5;
      value.metadata.tick = 100; value.texture = std::make_shared<texture_reference>();
      preservation_ticket ticket{value.id, 0, 0, value.texture};
      // A repeated before-clear opportunity can reuse an unread allocation in
      // this exact open recording; other sources/recordings and any GPU reader
      // require their own allocation. CPU ticket copies alone do not read it.
      if (!reusable_preserved(value, value.metadata, recording) ||
          reusable_preserved(value, value.metadata, recording + 1)) return false;
      value.consumers[0] = 30;
      if (reusable_preserved(value, value.metadata, recording)) return false;
      value.consumers[0] = 0;
      value.acquired = true;
      if (reusable_preserved(value, value.metadata, recording)) return false;
      value.acquired = false;
      // EOF copies can be consumed later in the same open recording. A ticket
      // on any other recording needs actual producer submission first.
      if (preserved_admission(value, ticket, consumer, device, recording) != consumer_status::ready ||
          preserved_admission(value, ticket, consumer, device, recording + 1) != consumer_status::capture_not_ready)
        return false;
      producer_submitted(value, producer, 7, true);
      if (reusable_preserved(value, value.metadata, recording) ||
          preserved_admission(value, ticket, consumer, device, recording + 1) != consumer_status::ready ||
          preserved_admission(value, ticket, consumer, device + 1, recording) != consumer_status::device_mismatch)
        return false;
      value.consumer_queue = consumer;
      if (preserved_admission(value, ticket, consumer + 1, device, recording) != consumer_status::ownership_mismatch)
        return false;
      std::array<bool, slot_limit> producers{}, consumers{}; producers[0] = true;
      if (!needs_submission_fence(producers, consumers)) return false; // Real snapshot, unlike a nomination.
      // Even matching API sequence/epoch/source metadata cannot turn a Generic
      // snapshot into API authority or make feature release destroy its lease.
      evaluations[0].epoch = 3; evaluations[0].sequence = 4; evaluations[0].source_id = 5;
      evaluations[0].views[0] = {3, 4, 100, 5, 0};
      queue_state owner; owner.device_identity = device;
      if (select_provider(owner, consumer, 1, 100, {}).value || select_provider(owner, consumer, 1, 100, {}).latest) return false;
      retire_source(provider_kind::streamline, 3, 5);
      if (value.invalid) return false;
      // Producer and consumer retirement still use the SAME queue-specific
      // obligations as API snapshots, not a frame-count or COM-only shortcut.
      consumer_submitted(value, consumer, 8, true);
      if (gpu_retired(value, [&](const fence_point &p) { return p.queue == producer && p.value <= 7; })) return false;
      if (!gpu_retired(value, [&](const fence_point &p) {
        return (p.queue == producer && p.value <= 7) || (p.queue == consumer && p.value <= 8);
      })) return false;
      invalidate(value, capture_failure::replay);
      if (preserved_admission(value, ticket, consumer, device, recording) != consumer_status::capture_not_ready) return false;
      // A CPU ticket keeps a completed/discarded slot out of the allocation
      // pool until the frame/command stats release it. A reused ID is rejected.
      value.invalid = false; value.producer_submitted = false; value.command = 0; value.consumers = {};
      value.retirements = {}; value.consumer_queue = 0;
      if (reclaimable(value)) return false;
      ++value.id;
      if (preserved_admission(value, ticket, consumer, device, recording) != consumer_status::ownership_mismatch) return false;
      ticket = {};
      return reclaimable(value);
    }
    bool submission_completion_regression() {
      auto available = std::find_if(slots.begin(), slots.end(), [](const auto &value) { return !value.id; });
      if (available == slots.end()) return false;
      struct cleanup {
        slot &value;
        slot saved;
        ~cleanup() { value = saved; }
      } restore{*available, *available};
      auto &value = *available;
      constexpr std::uint64_t native = 17, fence = 23;
      const auto restart = [&] { value = {}; value.id = ++serial; };
      // Drive the actual submission bookkeeping and acquire predicate, with
      // only the queue Signal result supplied by the CPU fixture.
      restart();
      producer_submitted(value, native, fence, true);
      if (value.invalid || value.finished || value.failure != capture_failure::none ||
          value.retirement_unknown || value.producer_fence != fence || submitted_success(value, native)) return false;
      finish(value.id, true);
      if (!submitted_success(value, native) || submitted_success(value, native + 1) || value.failure != capture_failure::none) return false;
      restart();
      finish(value.id, true);
      if (submitted_success(value, native)) return false;
      producer_submitted(value, native, fence, true);
      if (!submitted_success(value, native)) return false;
      // A failed result remains terminal in either ordering.
      for (const bool result_first : {false, true}) {
        restart();
        if (result_first) finish(value.id, false, capture_failure::evaluation_failed);
        producer_submitted(value, native, fence, true);
        if (!result_first) finish(value.id, false, capture_failure::evaluation_failed);
        if (!value.invalid || submitted_success(value, native) || value.failure != capture_failure::evaluation_failed) return false;
      }
      restart();
      producer_submitted(value, native, fence, false);
      finish(value.id, false);
      if (!value.invalid || !value.retirement_unknown || value.producer_fence ||
          submitted_success(value, native) || value.failure != capture_failure::producer_signal_failed) return false;
      restart();
      invalidate(value, capture_failure::observer_loss);
      finish(value.id, false);
      producer_submitted(value, native, fence, true);
      if (submitted_success(value, native) || value.failure != capture_failure::observer_loss) return false;
      restart();
      finish(value.id, true);
      producer_submitted(value, native, fence, true);
      producer_submitted(value, native + 1, fence + 1, true);
      return value.invalid && value.retirement_unknown && !submitted_success(value, native + 1) &&
        value.queue == native && value.producer_fence == fence &&
        value.failure == capture_failure::producer_queue_changed && cross_queue_completion_regression() &&
        preservation_owner_regression() && source_authority_regression();
    }
    bool snapshot_basis_regression() {
      // Provenance: admission still uses the mapped legacy state of an
      // enhanced layout, while the basis records where it came from.
      const auto saved = requested.load();
      struct restore_requested { bool value; ~restore_requested() { requested = value; } } restore{saved};
      requested = true;
      private_object list, target;
      const auto address = native_object(list);
      std::uint64_t cookie{};
      associate_recording(address, &cookie);
      auto owner = command(address, cookie, false);
      if (!cookie || !owner) return false;
      owner->reported = true;
      const native_observer::enhanced_texture shader_resource{reinterpret_cast<ID3D12Resource *>(&target),
        D3D12_BARRIER_LAYOUT_SHADER_RESOURCE, true};
      enhanced_textures(address, cookie, 1, &shader_resource);
      const auto identity = source_cookie(&target);
      const auto *entry = state(*owner, identity, false);
      if (!identity || !entry || !entry->known || entry->blocked || !entry->enhanced) return false;
      const auto chosen = copy_state::resolve(copy_state::rule::observed_else_contract, UINT32_MAX,
        copy_state::observation{true, entry->known, entry->blocked, entry->value}, 0);
      if (!chosen.admitted || !chosen.used_observed || copy_basis(entry, chosen) != state_basis::observed_enhanced) return false;
      // A later legacy transition replaces the provenance.
      D3D12_RESOURCE_BARRIER transition{};
      transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
      transition.Transition.pResource = reinterpret_cast<ID3D12Resource *>(&target);
      transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
      transition.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
      barriers(address, cookie, 1, &transition);
      entry = state(*owner, identity, false);
      if (!entry || entry->enhanced || copy_basis(entry, chosen) != state_basis::observed_legacy) return false;
      copy_state::decision contract{};
      contract.used_contract = true;
      if (copy_basis(nullptr, contract) != state_basis::contract || copy_basis(nullptr, {}) != state_basis::declared) return false;
      return std::string_view(name(state_basis::observed_enhanced)) == "observed_enhanced";
    }
  }
}

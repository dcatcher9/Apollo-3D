/*
 * Copyright (C) 2021 Patrick Mours
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Sunshine changes: bounded content-based scene depth selection and upscaler
 * eligibility, based on ReShade 6.8.0 example 09-depth, commit
 * 18deaa52de0c425a78b329e9cb3c497281cd00ec. Depth selection is always
 * automatic; the original manual depth controls were removed. This module
 * never reads another add-on's private C++ objects.
 */

// Generic depth's challenger probe, a part of depth_addon.cpp (included at
// its end, in the same translation unit): releasing the challenger,
// prepare_probe_target's choice and preservation of the probe target, the
// raw-sample selection key and completion, and sample_probe_target's
// sampling of the probe target or a due retained source. Declared in
// depth_addon.cpp.
#pragma once

static void release_challenger(effect_runtime *runtime, generic_depth_data &data, bool destroy_runtime = false)
{
	if (data.challenger_depth_stencil == 0)
		data.last_current_sample_tick = GetTickCount64();
	if (data.challenger_depth_stencil != 0)
	{
		if (auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>())
		{
			std::array<sunshine_depth_probe::capture_owner, 8> owners {};
			for (std::size_t i = 0; i != owners.size(); ++i)
				owners[i] = {data.member_views[i].source.handle, data.member_views[i].view.handle, data.member_views[i].retire_after};
			const bool retained = sunshine_depth_probe::capture_keeps_backup(data.challenger_depth_stencil.handle, owners);
			// The inherited capture path may have recorded work on other queues.
			// Retain its usual retirement interval before allocating another extra
			// backup, unless a nonretiring capture slot took over this allocation.
			// An in-flight sampler reference alone cannot skip that interval.
			// The released backup stays revivable for as long again, so the next
			// visit of the same shape reuses it instead of creating a full-size
			// texture on the present thread.
			constexpr auto gap = generic_depth_device_data::backup_retirement_frames;
			device_data->untrack_depth_stencil(runtime->get_device(), data.challenger_depth_stencil, 2 * gap);
			if (!retained) data.challenger_retired_until = device_data->frame_index + gap;
		}
		data.challenger_depth_stencil = { 0 };
		data.challenger_id = 0;
		data.challenger_samples = 0;
	}
	data.probe_target_id = 0;
	data.probe_started_tick = 0;
	if (destroy_runtime)
	{
		const std::lock_guard<std::mutex> guard(s_challenger_mutex);
		if (s_challenger_owner == runtime)
			s_challenger_owner = nullptr;
	}
}

static void prepare_probe_target(effect_runtime *runtime, command_list *cmd_list, generic_depth_data &data,
	generic_depth_device_data &device_data)
{
	if (!data.generic_capture_enabled)
		return;
	device *const device = runtime->get_device();
	if (data.challenger_depth_stencil == 0 && device_data.frame_index >= data.challenger_retired_until)
	{
		const std::lock_guard<std::mutex> guard(s_challenger_mutex);
		if (s_challenger_owner == runtime)
			s_challenger_owner = nullptr;
	}
	if (device->get_api() != device_api::d3d11 && device->get_api() != device_api::d3d12)
		return;

	if (data.challenger_depth_stencil == data.selected_depth_stencil && data.challenger_depth_stencil != 0)
		release_challenger(runtime, data); // The selected view now holds its own tracking reference.

	if (sunshine_depth::busy() || GetTickCount64() < data.next_sample_tick)
		return;

	// Missing/unsupported capture must not pin the round robin. A won comparison
	// keeps its visit until the takeover adopts the backup (begin_depth_frame).
	if (data.probe_target_id != 0 && sunshine_depth_probe::visit_expired(data.probe_started_tick, GetTickCount64(),
		data.challenger_id != 0 && data.selection.won_challenger_id() == data.challenger_id))
		release_challenger(runtime, data);
	if (data.probe_target_id == 0)
	{
		data.probe_target_id = data.discovery_hint.take(device_data.frame_index);
		if (data.probe_target_id && !data.selection.evidence(data.probe_target_id))
			data.probe_target_id = 0; // A recent hint cannot restore a destroyed or filtered candidate.
		// Refresh the current source between round-robin challenger visits so a
		// large candidate list cannot age out a working selection's evidence.
		if (!data.probe_target_id && GetTickCount64() - data.last_current_sample_tick >= 1000)
		{
			const std::shared_lock<std::shared_mutex> lock(s_mutex);
			if (const auto current = device_data.depth_stencil_resources.find(data.selected_depth_stencil);
				current != device_data.depth_stencil_resources.end() && current->second.last_used_in_frame == device_data.frame_index)
				data.probe_target_id = current->second.identity;
			data.last_current_sample_tick = GetTickCount64();
		}
		if (data.probe_target_id == 0)
		{
			// Discovery ranks against the logical anchor, which survives an
			// uncovered rotation turn when there is no current shader binding.
			data.probe_target_id = data.selection.next_probe(device_data.frame_index, data.anchor_identity);
			if (data.probe_target_id) data.discovery_hint.round_robin_started();
		}
		data.probe_started_tick = GetTickCount64();
	}
	if (data.probe_target_id == 0)
		return;

	resource target = { 0 };
	depth_stencil_resource info;
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		for (const auto &[candidate, candidate_info] : device_data.depth_stencil_resources)
		{
			if (candidate_info.identity == data.probe_target_id)
			{
				target = candidate;
				info = candidate_info;
				break;
			}
		}
	}
	if (target == 0)
	{
		release_challenger(runtime, data);
		return;
	}
	// Interleaved presents may render a different depth target. Keep this live
	// candidate's tracking reference through the selector's activity grace so
	// its next rendered frame can supply the remaining histogram samples.
	// Eligibility expiry and the probe timeout still release it; no inactive
	// source is copied or sampled as if it belonged to the current frame.
	if (info.last_used_in_frame != device_data.frame_index || !info.last_frame_stats.total.has_work())
		return;
	if (target == data.selected_depth_stencil)
		return; // Already prepared by the original selected-depth path.

	if (data.challenger_depth_stencil == 0)
	{
		if (device_data.frame_index < data.challenger_retired_until)
			return;
		{
			const std::lock_guard<std::mutex> guard(s_challenger_mutex);
			if (s_challenger_owner != nullptr && s_challenger_owner != runtime)
				return;
			s_challenger_owner = runtime;
		}
		depth_stencil_backup *const backup = device_data.track_depth_stencil_for_backup(device, target, info.desc);
		if (backup == nullptr)
		{
			data.probe_target_id = 0;
			return;
		}
		data.challenger_depth_stencil = target;
		data.challenger_id = info.identity;
		data.challenger_started_frame = device_data.frame_index;
		// Retirement wait is not measurement time. Both the queued visit and
		// this allocated capture phase have their own bounded two-second clock.
		data.probe_started_tick = GetTickCount64();
		data.challenger_samples = 0;
		return; // Let the challenger render one full frame before its first end-of-frame copy.
	}
	if (data.challenger_depth_stencil != target || data.challenger_id != info.identity)
	{
		release_challenger(runtime, data);
		return;
	}

	depth_stencil_backup *const backup = device_data.find_depth_stencil_backup(target);
	if (backup == nullptr || backup->backup_texture == 0)
		return;
	if (info.last_frame_stats.copied_during_frame)
	{
		// This frame's end-of-frame copy (a capture member's or another
		// runtime's) already holds the D3D12 pixels in the common owner's
		// snapshot. Unqualified challengers have no member/selected
		// publication path yet; publish that snapshot so the probe can
		// establish quality.
		if (device->get_api() == device_api::d3d12)
			preserve_depth_at_effects(runtime, cmd_list, device_data, target, info.identity, data.challenger_started_frame);
		return;
	}

	if ((info.desc.usage & (resource_usage::copy_source | resource_usage::resolve_source)) != 0)
		preserve_depth_at_effects(runtime, cmd_list, device_data, target, info.identity, data.challenger_started_frame);
	else if (device_data.frame_index > data.challenger_started_frame + 120)
	{
		// No copy is missing data, not proof of a flat source. Move on so a single
		// incompatible buffer cannot stop probing other low-draw candidates.
		release_challenger(runtime, data);
	}
}

static sunshine_camera_binding::selection_key raw_selection_key(effect_runtime *runtime, const generic_depth_data &data,
	const generic_depth_device_data &device_data, resource target, const depth_stencil_resource &info,
	const depth_stencil_backup &backup, const resource_desc &backup_desc)
{
	const auto region = effective_capture_region(info);
	sunshine_camera_binding::selection_key key;
	key.scope = { reinterpret_cast<uint64_t>(runtime), data.runtime_epoch, runtime->get_device()->get_native(), device_data.device_epoch };
	key.original = { target.handle, info.identity, 0, info.desc.texture.width, info.desc.texture.height,
		region.x, region.y, region.width, region.height };
	key.sampled = { backup.backup_texture.handle, backup.content_identity };
	key.sample_format = static_cast<uint32_t>(backup_desc.texture.format);
	key.sample_width = backup_desc.texture.width;
	key.sample_height = backup_desc.texture.height;
	key.crop = { region.x, region.y, region.width, region.height };
	key.layout_epoch = info.layout_epoch;
	return key;
}

static void complete_raw_sample(effect_runtime *runtime, generic_depth_data &data,
	generic_depth_device_data &device_data, const sunshine_depth::sample_result &completed)
{
	if (completed.capture_id == 0 || completed.capture_id != data.raw_pending_capture_id)
		return;
	const auto cancel = [&data]() {
		data.raw_sample_binding.cancel();
		data.raw_pending_capture_id = 0;
	};
	if (!data.raw_scene_requested || data.raw_pending_metadata.basis_epoch != data.raw_basis_epoch)
	{
		cancel();
		return;
	}
	depth_stencil_resource info;
	bool found_source = false;
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		const auto found = device_data.depth_stencil_resources.find(data.sampled_depth_stencil);
		if (found != device_data.depth_stencil_resources.end() && found->second.identity == completed.token)
		{
			info = found->second;
			found_source = true;
		}
	}
	const auto *backup = device_data.find_depth_stencil_backup(data.sampled_depth_stencil);
	if (!found_source || backup == nullptr || backup->backup_texture == 0)
	{
		cancel();
		return;
	}
	// Native/SDK resource calls stay outside the depth-resource map mutex.
	const auto backup_desc = runtime->get_device()->get_resource_desc(backup->backup_texture);
	const auto current = raw_selection_key(runtime, data, device_data, data.sampled_depth_stencil, info, *backup, backup_desc);
	if (data.raw_sample_binding.invalidate_if_changed(current) != sunshine_camera_binding::status::unchanged)
	{
		data.raw_pending_capture_id = 0;
		return;
	}
	sunshine_camera_binding::readback readback;
	readback.capture_id = completed.capture_id;
	readback.scope = completed.capture_scope;
	readback.sampled = { completed.source.handle, completed.source_lifetime };
	readback.source_format = completed.source_format;
	readback.source_width = completed.source_width;
	readback.source_height = completed.source_height;
	readback.crop = { completed.viewport_x, completed.viewport_y, completed.viewport_width, completed.viewport_height };
	readback.width = completed.width;
	readback.height = completed.height;
	readback.values = completed.values.data();
	readback.value_count = completed.values.size();
	readback.valid = completed.valid;
	sunshine_camera_binding::associated_sample associated;
	const auto status = data.raw_sample_binding.complete(readback, GetTickCount64(), associated);
	data.raw_pending_capture_id = 0;
	if (status != sunshine_camera_binding::status::associated || !data.raw_pending_metadata.depth_ready ||
		!completed.moments.supplied)
		return; // Selector-only point samples cannot become geometry evidence after promotion.
	// This source was retained when the full-image measurement was requested.
	// Sparse selector points can miss its geometry; their classification chooses
	// candidates but cannot veto an authenticated full-image measurement here.
	// Deliver a requested but malformed full-image measurement too: the scene
	// policy must invalidate it rather than keep learning from older evidence.
	data.raw_latest_submission = associated.captured;
	data.raw_latest_sample = {};
	data.raw_latest_sample.id = associated.sample.id;
	data.raw_latest_sample.capture_ms = associated.sample.capture_ms;
	data.raw_latest_sample.metadata = data.raw_pending_metadata;
	data.raw_latest_sample.readback_frame = data.raw_pending_metadata.frame;
	data.raw_latest_sample.readback_source = data.raw_pending_metadata.source;
	data.raw_latest_sample.readback_layout_epoch = associated.captured.selection.layout_epoch;
	// Geometry consumes the full-image moments; dynamic selector points are not
	// squeezed into the historical fixed-grid fixture carried by scene::sample.
	data.raw_latest_sample.range_supplied = true;
	data.raw_latest_sample.range_valid = completed.range_valid;
	data.raw_latest_sample.range_min = completed.range_min;
	data.raw_latest_sample.range_max = completed.range_max;
	data.raw_latest_sample.moments = completed.moments;
	data.raw_latest_available = true;
	// The asynchronous result belongs to its submitted physical member, even
	// when another source is the currently preferred rendering allocation.
	if (auto *member = raw_member(data, data.raw_pending_metadata.source.lifetime))
	{
		member->sample = data.raw_latest_sample;
		member->submission = associated.captured;
		member->available = true;
	}
}

static void sample_probe_target(effect_runtime *runtime, command_list *cmd_list, generic_depth_data &data)
{
	if (!data.generic_capture_enabled)
		return;
	const uint64_t now = GetTickCount64();
	// Every retained source owns its deadline. Sampling a usable off-turn copy
	// does not depend on which other copy the renderer currently prefers.
	generic_depth_data::raw_member_sample *due_member = nullptr;
	if (data.raw_scene_requested)
		for (unsigned i = 0; i != data.captures.count; ++i)
		{
			const auto &record = data.captures.records[i];
			auto *member = raw_member(data, record.identity);
			if (member && now >= member->next_tick && record.current(data.captures.present, data.captures.frame, data.captures.runtime) &&
				(!due_member || member->next_tick < due_member->next_tick)) due_member = member;
		}
	const bool raw_selected_due = due_member != nullptr;
	const uint64_t target_id = due_member ? due_member->identity : data.probe_target_id;
	if (target_id == 0 || sunshine_depth::busy() || (!raw_selected_due && now < data.next_sample_tick))
		return;
	device *const device = runtime->get_device();
	auto *device_data = device->get_private_data<generic_depth_device_data>();
	if (device_data == nullptr)
		return;
	resource target = { 0 };
	depth_stencil_resource info;
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		for (const auto &[candidate, candidate_info] : device_data->depth_stencil_resources)
		{
			if (candidate_info.identity == target_id && candidate_info.last_used_in_frame == device_data->frame_index &&
				candidate_info.last_frame_stats.total.has_work())
			{
				target = candidate;
				info = candidate_info;
				break;
			}
		}
	}
	if (target == 0) return;
	uint64_t capture_after = 0;
	bool retained = false;
	for (const auto &member : data.member_views)
		if (member.identity == target_id && member.view != 0 && !member.retire_after)
		{ capture_after = member.capture_after; retained = true; break; }
	if (!retained && target != data.selected_depth_stencil && (target != data.challenger_depth_stencil ||
		device_data->frame_index <= data.challenger_started_frame)) return;
	const auto captured = capture_record(runtime, data, *device_data, target, info.identity, {}, true, capture_after);
	if (!captured.record.current(device_data->native_present_index, device_data->frame_index, data.runtime_epoch)) return;

	depth_stencil_backup *const backup = device_data->find_depth_stencil_backup(target);
	if (backup == nullptr || backup->backup_texture == 0)
		return; // Direct-access D3D11 sources are left untouched until they have a backup.
	if (backup->backup_texture.handle != captured.record.sampled || backup->content_identity != captured.record.assignment ||
		info.layout_epoch != captured.record.layout) return;
	sunshine_depth::sample_request request;
	request.collect_range = data.raw_scene_requested;
	request.token = info.identity;
	request.source = backup->backup_texture;
	request.before = resource_usage::copy_dest;
	request.source_width = captured.metadata.width;
	request.source_height = captured.metadata.height;
	const capture_region region {captured.metadata.x, captured.metadata.y, captured.metadata.active_width,
		captured.metadata.active_height, captured.metadata.layout_epoch};
	request.x = region.x;
	request.y = region.y;
	request.width = region.width;
	request.height = region.height;
	if (data.raw_scene_requested)
	{
		// Freeze owner/source/copy facts now. Neither completion nor the exporter
		// is allowed to retrofit a newer camera, orientation, frame or basis.
		const auto backup_desc = device->get_resource_desc(backup->backup_texture);
		sunshine_camera_binding::submission binding;
		binding.selection = raw_selection_key(runtime, data, *device_data, target, info, *backup, backup_desc);
		binding.selection.original = {captured.record.source, captured.record.identity, 0, captured.metadata.width, captured.metadata.height,
			region.x, region.y, region.width, region.height};
		binding.selection.crop = {region.x, region.y, region.width, region.height};
		binding.selection.layout_epoch = captured.record.layout;
		const auto grid = sunshine_depth_statistics::tile_grid(region.width, region.height);
		binding.grid_width = grid.x;
		binding.grid_height = grid.y;
		binding.camera.unit_epoch = data.raw_basis_epoch;
		binding.camera.frame = { captured.record.frame, captured.record.runtime };
		binding.camera.source = binding.selection.original;
		binding.projection_associated = false;
		binding.capture_frame = captured.record.frame;
		binding.capture_ms = now;
		data.raw_sample_binding.cancel(); // Sampler is idle; no valid old slot remains.
		if (data.raw_sample_binding.begin(binding, now, request.capture_id) == sunshine_camera_binding::status::submitted)
		{
			request.capture_scope = binding.selection.scope;
			request.source_lifetime = backup->content_identity;
			data.raw_pending_capture_id = request.capture_id;
			data.raw_pending_metadata = {};
			data.raw_pending_metadata.basis_epoch = data.raw_basis_epoch;
			data.raw_pending_metadata.layout_epoch = info.layout_epoch;
			data.raw_pending_metadata.source = binding.selection.original;
			data.raw_pending_metadata.frame = binding.camera.frame;
			data.raw_pending_metadata.direction = captured.metadata.detected_orientation;
			// Raw geometry and its full-image moments must use the same frozen
			// convention. Unknown direction still permits selector point sampling.
			const auto direction = data.raw_pending_metadata.direction;
			if (raw_member(data, target_id) != nullptr &&
				(direction == sunshine_depth::depth_orientation::normal || direction == sunshine_depth::depth_orientation::reversed))
			{
				request.collect_moments = true;
				request.moments_A = direction == sunshine_depth::depth_orientation::normal ? 1.f : 0.f;
				request.moments_inverseB = direction == sunshine_depth::depth_orientation::normal ? -1.f : 1.f;
			}
			data.raw_pending_metadata.depth_ready = captured.metadata.ready;
			data.raw_pending_metadata.copy_ambiguous = captured.record.ambiguous;
			uint32_t frame_width = 0, frame_height = 0;
			runtime->get_screenshot_width_and_height(&frame_width, &frame_height);
			data.raw_pending_metadata.aligned_viewport_assumed = raw_alignment_assumed(region,
				info.desc.texture.width, info.desc.texture.height, frame_width, frame_height);
		}
		else data.raw_pending_capture_id = 0;
	}

	// The native sampler retains its COM objects too. This tracking reference
	// additionally prevents the original backup pool from repurposing its memory.
	device_data->track_depth_stencil_for_backup(device, target, info.desc);
	data.sampled_depth_stencil = target;
	data.sampled_frame = device_data->frame_index;
	data.sampled_region = region;
	// Selected raw sampling cannot be starved by a long challenger round-robin.
	// With this opt-in, leave intervening slots available to the selector too.
	// A rotating group can refresh a different raw member every few frames.
	// Those independent deadlines must not continually postpone challenger work.
	if (!raw_selected_due || target_id == data.probe_target_id)
		data.next_sample_tick = now + (data.raw_scene_requested ? 125 : 250);
	if (data.raw_scene_requested)
		if (auto *member = raw_member(data, target_id)) member->next_tick = now + 250;
	if (!sunshine_depth::submit(runtime, cmd_list, request))
	{
		data.raw_sample_binding.cancel();
		data.raw_pending_capture_id = 0;
		release_sample_reference(runtime, data);
		if (target != data.challenger_depth_stencil || ++data.challenger_samples >= 3)
			release_challenger(runtime, data);
	}
}

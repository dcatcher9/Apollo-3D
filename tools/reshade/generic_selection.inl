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

// Generic depth's candidate selection, a part of depth_addon.cpp (included at
// its end, in the same translation unit): the aspect, capture-region and shape
// rules a depth-stencil must meet, the per-candidate observations, and
// select_depth_candidate, begin_depth_frame's selection of this Present's
// source. Declared in depth_addon.cpp.
#pragma once

// Checks whether the aspect ratio of the two sets of dimensions is similar or not
static bool check_aspect_ratio(float width_to_check, float height_to_check, float width, float height)
{
	if (width_to_check == 0.0f || height_to_check == 0.0f)
		return true;

	const float w_ratio = width / width_to_check;
	const float h_ratio = height / height_to_check;
	const float aspect_ratio_delta = (width / height) - (width_to_check / height_to_check);

	// Upscalers render scene depth below output resolution; DLSS Performance and
	// Ultra Performance are outside the original 1.85 output/input ratio limit.
	// The original float test below stays separate: at the 0.1 aspect limit its
	// rounding admits sizes the double test rejects (test_depth_addon_exports.cpp).
	if (sunshine_depth::auto_candidate_shape(static_cast<uint32_t>(width_to_check), static_cast<uint32_t>(height_to_check),
			static_cast<uint32_t>(width), static_cast<uint32_t>(height)))
		return true;

	// Accept if dimensions are similar in value
	return std::abs(aspect_ratio_delta) <= 0.1f && w_ratio <= 1.85f && w_ratio >= 0.5f && h_ratio <= 1.85f && h_ratio >= 0.5f;
}

static bool valid_depth_viewport(const viewport &view, const resource_desc &desc)
{
	return std::isfinite(view.x) && std::isfinite(view.y) && std::isfinite(view.width) && std::isfinite(view.height) &&
		view.x >= 0 && view.y >= 0 && view.width >= 1 && view.height >= 1 &&
		view.x + view.width <= desc.texture.width + 0.5f && view.y + view.height <= desc.texture.height + 0.5f;
}

template <typename DepthInfo>
SUNSHINE_DEPTH_ADDON_INTERNAL capture_region effective_capture_region(const DepthInfo &info)
{
	if (info.have_preserved_region) return info.preserved_region;
	const viewport &view = info.last_frame_stats.copied_during_frame ?
		info.last_frame_stats.copied_viewport : info.last_frame_stats.total.last_viewport;
	capture_region result { 0, 0, info.desc.texture.width, info.desc.texture.height, info.layout_epoch };
	if (valid_depth_viewport(view, info.desc))
	{
		result.x = static_cast<uint32_t>(view.x);
		result.y = static_cast<uint32_t>(view.y);
		result.width = static_cast<uint32_t>(view.width);
		result.height = static_cast<uint32_t>(view.height);
	}
	return result;
}

static bool raw_alignment_assumed(const capture_region &region, uint32_t source_width, uint32_t source_height,
	uint32_t color_width, uint32_t color_height)
{
	return source_width && source_height && color_width && color_height && region.x == 0 && region.y == 0 &&
		region.width == source_width && region.height == source_height &&
		std::abs(static_cast<float>(source_width) / source_height - static_cast<float>(color_width) / color_height) <= 0.1f &&
		check_aspect_ratio(static_cast<float>(region.width), static_cast<float>(region.height),
			static_cast<float>(color_width), static_cast<float>(color_height));
}

template <typename DepthInfo>
SUNSHINE_DEPTH_ADDON_INTERNAL void observe_candidate(generic_depth_data &data, const DepthInfo &info, uint32_t width, uint32_t height)
{
	capture_region region = effective_capture_region(info);
	const auto previous = data.observed_regions.find(info.identity);
	if ((!info.last_frame_stats.copied_during_frame || info.last_frame_stats.copy_provenance_ambiguous) &&
		previous != data.observed_regions.end() && previous->second.layout_epoch == info.layout_epoch)
		region = previous->second; // Untracking a probe does not prove its captured scene region changed.
	if (previous != data.observed_regions.end() && !(previous->second == region))
	{
		data.selection.erase(info.identity); // Content crop changes do not recreate the tracked resource.
		data.native_calibrations.erase(info.identity);
	}
	data.observed_regions[info.identity] = region;
	data.selection.observe({ info.identity, info.desc.texture.width, info.desc.texture.height,
		static_cast<float>(region.width), static_cast<float>(region.height), width, height, info.last_used_in_frame,
		info.last_frame_stats.total.work_count(), info.last_frame_stats.total.vertices });
}

SUNSHINE_DEPTH_ADDON_INTERNAL void prune_depth_selection(generic_depth_data &data, const std::vector<uint64_t> &eligible_ids)
{
	for (const uint64_t previous_id : data.observed_ids)
		if (std::find(eligible_ids.begin(), eligible_ids.end(), previous_id) == eligible_ids.end())
		{
			data.selection.erase(previous_id);
			data.observed_regions.erase(previous_id);
			data.native_calibrations.erase(previous_id);
		}
	data.observed_ids = eligible_ids;
}

SUNSHINE_DEPTH_ADDON_INTERNAL void observe_preserved_layout(depth_stencil_resource &info, const depth_stencil_frame_stats &stats)
{
	if (!stats.copied_during_frame || stats.copy_provenance_ambiguous) return;
	capture_region region {0, 0, info.desc.texture.width, info.desc.texture.height, info.layout_epoch};
	if (valid_depth_viewport(stats.copied_viewport, info.desc))
	{
		region.x = static_cast<uint32_t>(stats.copied_viewport.x);
		region.y = static_cast<uint32_t>(stats.copied_viewport.y);
		region.width = static_cast<uint32_t>(stats.copied_viewport.width);
		region.height = static_cast<uint32_t>(stats.copied_viewport.height);
	}
	if (info.have_preserved_region && !(info.preserved_region == region))
	{
		region.layout_epoch = ++info.layout_epoch;
		info.orientation_evidence = {};
	}
	info.preserved_region = region;
	info.have_preserved_region = true;
}

template <typename DepthInfo>
SUNSHINE_DEPTH_ADDON_INTERNAL bool candidate_matches_shape(const DepthInfo &info, uint32_t width, uint32_t height)
{
	if (check_aspect_ratio(static_cast<float>(info.desc.texture.width), static_cast<float>(info.desc.texture.height),
		static_cast<float>(width), static_cast<float>(height)))
		return true;

	// Padded allocations may contain a smaller active DLSS/FSR viewport; its
	// captured region qualifies when the whole allocation does not.
	const auto region = effective_capture_region(info);
	return check_aspect_ratio(static_cast<float>(region.width), static_cast<float>(region.height), static_cast<float>(width), static_cast<float>(height));
}

// The selector's choice for one Present (selection_policy::choose). A
// challenger takes over only on a Present that advanced the depth frame with
// its own work, and only with a backup already tracked for it (the probe's,
// or a capture slot's), so its first anchored Present has its own current
// copy: a new capture slot counts copies only from the next Present.
SUNSHINE_DEPTH_ADDON_INTERNAL sunshine_depth::selection_result choose_generic_source(sunshine_depth::selection_policy &selection,
	generic_depth_device_data &device_data, const depth_selection_snapshot &snapshot, uint64_t current_id, bool depth_present)
{
	return selection.choose(device_data.frame_index, current_id, depth_present, [&](uint64_t id) {
		for (const auto &[candidate, info] : snapshot)
			if (info.identity == id) return device_data.find_depth_stencil_backup(candidate) != nullptr;
		return false;
	});
}

// begin_depth_frame's candidate selection for one Present: snapshots the
// device's depth-stencils under the shared s_mutex, observes and prunes the
// eligible candidates, consumes the completed sample, chooses this Present's
// source and settles the probe. Its parameters are begin_depth_frame's
// locals under their names, inputs first; it assigns every output (selected
// .. any_warming) in the order begin_depth_frame declared them.
static void select_depth_candidate(effect_runtime *runtime, device *const device,
	generic_depth_device_data *const device_data, generic_depth_data &data, const bool automatic,
	const std::optional<sunshine_depth::sample_result> &completed_sample,
	const sunshine_depth::depth_quality &completed_quality, const bool completed_other_raw_sample,
	resource &selected_depth_stencil, const depth_selection_resource *&selected_depth_stencil_info,
	uint32_t &frame_width, uint32_t &frame_height, uint64_t &native_present_index,
	depth_selection_snapshot &current_depth_stencil_resources,
	bool &any_recent_activity, bool &any_filtered, bool &any_warming)
{
	selected_depth_stencil = { 0 };
	selected_depth_stencil_info = nullptr;

	runtime->get_screenshot_width_and_height(&frame_width, &frame_height);

	std::shared_lock<std::shared_mutex> lock(s_mutex);
	native_present_index = device_data->native_present_index;
	if (s_frame_activity_trace)
	{
		data.trace_snapshot_present = native_present_index;
		data.trace_snapshot_frame = device_data->frame_index;
	}
	current_depth_stencil_resources = snapshot_depth_selection(device_data->depth_stencil_resources);
	// Unlock before calling into device below, since device may hold a lock itself and that then can deadlock another thread that calls into 'on_destroy_resource' from the device holding that lock
	lock.unlock();

	uint64_t current_id = 0;
	const auto policy_source = data.anchor_identity ? data.anchor_depth_stencil : data.selected_depth_stencil;
	const auto policy_identity = data.anchor_identity ? data.anchor_identity : data.selected_identity;
	if (const auto current = find_depth_selection(current_depth_stencil_resources, policy_source);
		current != current_depth_stencil_resources.end() && current->second.identity == policy_identity)
		current_id = current->second.identity;

	std::vector<uint64_t> eligible_ids;
	any_recent_activity = false;
	any_filtered = false;
	any_warming = false;
	for (auto &[depth_stencil, info] : current_depth_stencil_resources)
	{
		if (info.last_used_in_frame != std::numeric_limits<uint64_t>::max() && info.last_used_in_frame + 3 >= device_data->frame_index)
			any_recent_activity = true;
		if (info.first_used_in_frame != std::numeric_limits<uint64_t>::max() && device_data->frame_index <= info.first_used_in_frame + 1)
			any_warming = true;

		if (info.desc.texture.samples > 1 && !device->check_capability(device_caps::resolve_depth_stencil))
		{
			any_filtered = true;
			continue; // Ignore multisampled textures, since they would need to be resolved first
		}

		if (!candidate_matches_shape(info, frame_width, frame_height))
		{
			any_filtered = true;
			continue; // Not a good fit
		}

		// Inactivity affects current preference, not the lifetime of content
		// evidence or completed numeric observations for a retained allocation.
		if (automatic && info.first_used_in_frame != std::numeric_limits<uint64_t>::max())
		{
			eligible_ids.push_back(info.identity);
			observe_candidate(data, info, frame_width, frame_height);
		}
		if (!info.last_frame_stats.total.has_work() || (!automatic && info.last_frame_stats.total.vertices <= 3 && info.last_frame_stats.total.drawcalls_indirect == 0 && info.last_frame_stats.total.scene_writes == 0))
			continue; // Skip unused

		if (info.last_used_in_frame == std::numeric_limits<uint64_t>::max() ||
			info.first_used_in_frame == std::numeric_limits<uint64_t>::max() ||
			info.last_used_in_frame + (automatic ? 3 : 0) < device_data->frame_index || device_data->frame_index <= (info.first_used_in_frame + 1))
			continue; // Content policy preserves its short activity grace; fresh resources still warm up.

		if (info.last_used_in_frame < device_data->frame_index)
			continue;

		if (selected_depth_stencil.handle == 0 ||
			info.last_frame_stats.total > selected_depth_stencil_info->last_frame_stats.total)
		{
			selected_depth_stencil = depth_stencil;
			selected_depth_stencil_info = &info;
		}
	}
	if (!automatic && (data.native_depth_requested || data.raw_scene_requested) && selected_depth_stencil_info != nullptr)
	{
		// Other APIs keep the original statistics choice, but its native
		// calibration still needs current capture-region metadata. Observe only
		// that choice; never admit a challenger.
		eligible_ids.push_back(selected_depth_stencil_info->identity);
		observe_candidate(data, *selected_depth_stencil_info, frame_width, frame_height);
	}
	// Every eligible ID came from this live snapshot. This one pass therefore
	// removes destroyed and filtered candidates before consuming readbacks or
	// choosing a source.
	prune_depth_selection(data, eligible_ids);

	if (completed_sample)
	{
		const auto region = data.observed_regions.find(completed_sample->token);
		const bool region_matches = region != data.observed_regions.end() && region->second == data.sampled_region &&
			completed_sample->viewport_x == region->second.x && completed_sample->viewport_y == region->second.y &&
			completed_sample->viewport_width == region->second.width && completed_sample->viewport_height == region->second.height;
		if (completed_sample->valid && region_matches)
		{
			data.selection.sample(completed_sample->token, completed_quality, data.sampled_frame);
			// Only the independent reference renderer consumes this calibration.
			if (data.native_depth_requested)
			{
				if (data.native_calibrations.find(completed_sample->token) == data.native_calibrations.end() && data.native_calibrations.size() >= 64)
				{
					const auto oldest = std::min_element(data.native_calibrations.begin(), data.native_calibrations.end(),
						[](const auto &a, const auto &b) { return a.second.last_sample_frame() < b.second.last_sample_frame(); });
					data.native_calibrations.erase(oldest);
				}
				const sunshine_depth::depth_calibration_key key { completed_sample->token, region->second.layout_epoch,
					completed_sample->source_width, completed_sample->source_height,
					region->second.x, region->second.y, region->second.width, region->second.height };
				data.native_calibrations[completed_sample->token].sample(key, data.sampled_frame,
					completed_sample->values.data(), completed_sample->values.size(), completed_quality.kind == sunshine_depth::content_kind::useful);
			}
		}
	}

	const char *selection_reason = "draw statistics";
	sunshine_depth::selection_result choice;
	if (automatic)
	{
		choice = choose_generic_source(data.selection, *device_data, current_depth_stencil_resources, current_id,
			device_data->native_depth_present_index == native_present_index);
		for (const auto &[candidate, info] : current_depth_stencil_resources)
		{
			if (info.identity == choice.id || (!choice.id && info.identity == current_id &&
				std::find(eligible_ids.begin(), eligible_ids.end(), current_id) != eligible_ids.end()))
			{
				selected_depth_stencil = candidate;
				selected_depth_stencil_info = &info;
				selection_reason = choice.id ? choice.reason : "retained capture budget; waiting for current depth";
				break;
			}
		}
	}

	if (completed_sample)
	{
		// The first three positive captures only qualify a challenger. Keep its
		// existing backup while fresh paired wins finish, rather than restarting
		// after a full round through unrelated sources. Do not reset the original
		// probe deadline: an incomplete comparison must still yield to others.
		// Automatic raw sampling independently refreshes the incumbent while
		// this probe holds the challenger. Other modes retain their old cadence.
		// A won comparison keeps the visit until the takeover, which adopts its
		// backup with the takeover Present's own copy (select_captured_depth);
		// prepare_probe_target then releases the probe, as for any selected
		// source.
		const bool comparison_pending = automatic && data.raw_scene_requested &&
			choice.id == current_id && data.selection.pending_challenger_id() == data.challenger_id;
		const bool won = automatic && data.challenger_id != 0 && data.selection.won_challenger_id() == data.challenger_id;
		const bool own = completed_sample->token == data.challenger_id;
		if (own) ++data.challenger_samples;
		if (!sunshine_depth_probe::keep_probe_after_sample(own, data.challenger_samples, comparison_pending, won,
			data.raw_scene_requested && completed_other_raw_sample, completed_sample->token, data.probe_target_id))
			release_challenger(runtime, data);
		else if (own)
			data.probe_target_id = data.challenger_id;
	}

	data.selection_reason = selection_reason;

	if (!automatic || (data.probe_target_id != 0 &&
		std::find(eligible_ids.begin(), eligible_ids.end(), data.probe_target_id) == eligible_ids.end()))
		release_challenger(runtime, data);
}

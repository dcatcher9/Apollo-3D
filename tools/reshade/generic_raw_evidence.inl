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

// Generic depth's raw-scene evidence, a part of depth_addon.cpp (included at
// its end, in the same translation unit): the exporter's raw-scene request,
// the current raw samples and source roster, and the validation of raw-scene
// history against the live depth-stencil allocations. Declared in
// depth_addon.h.
#pragma once

void sunshine_depth::set_raw_scene_request(effect_runtime *runtime, uint64_t basis_epoch, bool enabled)
{
	if (!s_registered || runtime == nullptr) return;
	auto *data = runtime->get_private_data<generic_depth_data>();
	if (data == nullptr) return;
	if (!enabled || basis_epoch == 0)
	{
		if (data->raw_scene_requested || data->raw_pending_capture_id || data->raw_latest_available)
			clear_raw_sample_request(*data);
		return;
	}
	if (!data->raw_scene_requested || data->raw_basis_epoch != basis_epoch)
	{
		clear_raw_sample_request(*data);
		data->raw_scene_requested = true;
		data->raw_basis_epoch = basis_epoch;
	}
}

std::size_t sunshine_depth::get_raw_samples(effect_runtime *runtime,
	std::array<sunshine_raw_scene::sample, sunshine_raw_scene::maximum_sources> &output)
{
	output = {};
	if (!s_registered || runtime == nullptr) return 0;
	const auto *data = runtime->get_private_data<generic_depth_data>();
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (!data || !device_data || !data->raw_scene_requested) return 0;
	std::size_t count = 0;
	const auto now = GetTickCount64();
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	for (const auto &member : data->raw_members)
	{
		if (!member.available) continue;
		const auto &sample = member.sample;
		if (sample.metadata.basis_epoch != data->raw_basis_epoch || sample.metadata.frame.token_generation != data->runtime_epoch ||
			sample.metadata.frame.frame > device_data->frame_index || now < sample.capture_ms ||
			now - sample.capture_ms >= sunshine_camera_binding::maximum_age_ms) continue;
		const auto found = device_data->depth_stencil_resources.find(resource{sample.metadata.source.native});
		if (found == device_data->depth_stencil_resources.end() || found->second.identity != sample.metadata.source.lifetime) continue;
		const auto &info = found->second;
		const auto region = effective_capture_region(info);
		if (info.layout_epoch != sample.metadata.layout_epoch || info.desc.texture.width != sample.metadata.source.width ||
			info.desc.texture.height != sample.metadata.source.height || region.x != sample.metadata.source.left ||
			region.y != sample.metadata.source.top || region.width != sample.metadata.source.extent_width ||
			region.height != sample.metadata.source.extent_height || info.orientation_evidence.detected() != sample.metadata.direction) continue;
		// The completed CPU payload already authenticated its immutable submission
		// and backup assignment. Subsequent GPU storage retirement does not alter it.
		output[count++] = sample;
	}
	return count;
}

bool sunshine_depth::get_raw_source_roster(effect_runtime *runtime, raw_source_roster &output)
{
	output = {};
	if (!s_registered || runtime == nullptr) return false;
	const auto *data = runtime->get_private_data<generic_depth_data>();
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (!data || !device_data || !data->raw_scene_requested) return false;
	output.routing_epoch = data->raw_roster.routing_epoch;
	output.basis_epoch = data->raw_basis_epoch; // Exporter can change this after the selector callback.
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	output.observed_frame = {device_data->frame_index, data->runtime_epoch};
	for (unsigned i = 0; i != data->raw_roster.count; ++i)
	{
		const auto &member = data->raw_roster.members[i];
		const auto found = device_data->depth_stencil_resources.find(resource{member.source.native});
		if (found == device_data->depth_stencil_resources.end()) continue;
		const auto &info = found->second;
		const auto region = effective_capture_region(info);
		if (info.identity != member.source.lifetime || info.layout_epoch != member.layout_epoch ||
			info.desc.texture.width != member.source.width || info.desc.texture.height != member.source.height ||
			region.x != member.source.left || region.y != member.source.top ||
			region.width != member.source.extent_width || region.height != member.source.extent_height) continue;
		output.members[output.count] = member;
		output.members[output.count++].direction = info.orientation_evidence.detected();
	}
	return output.routing_epoch != 0;
}

std::uint32_t sunshine_depth::validate_raw_history(effect_runtime *runtime,
	const std::array<sunshine_raw_scene::source_basis, sunshine_raw_scene::maximum_history_sources> &history)
{
	static_assert(sunshine_raw_scene::maximum_history_sources <= 32);
	if (!s_registered || runtime == nullptr) return 0;
	const auto *data = runtime->get_private_data<generic_depth_data>();
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (!data || !device_data || !data->raw_scene_requested) return 0;
	std::uint32_t live = 0;
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	for (std::size_t i = 0; i != history.size(); ++i)
	{
		const auto &member = history[i];
		if (!member.source.native || member.source.viewport != 0 || member.convention_epoch != 0) continue;
		const auto found = device_data->depth_stencil_resources.find(resource{member.source.native});
		if (found == device_data->depth_stencil_resources.end()) continue;
		const auto &info = found->second;
		const auto region = effective_capture_region(info);
		if (info.identity != member.source.lifetime || info.layout_epoch != member.layout_epoch ||
			info.desc.texture.width != member.source.width || info.desc.texture.height != member.source.height ||
			region.x != member.source.left || region.y != member.source.top ||
			region.width != member.source.extent_width || region.height != member.source.extent_height ||
			info.orientation_evidence.detected() != member.direction) continue;
		// Numerical history belongs to the live original allocation. A different
		// source owning the current GPU backup slot neither validates nor erases it.
		live |= std::uint32_t{1} << i;
	}
	return live;
}

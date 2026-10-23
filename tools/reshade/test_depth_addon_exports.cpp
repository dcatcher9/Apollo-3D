/*
 * Copyright (C) 2021 Patrick Mours
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Sunshine test add-on: depth_addon.cpp's runtime-test exports and its
 * depth-selection snapshot contract, moved from that file. Built only into
 * SunshineSBSTest; they reach its internals through depth_addon_test_access.h.
 */

#ifndef SUNSHINE_SBS_RUNTIME_TEST_ADDON
#error "Built only into the test add-on (SunshineSBSTest)"
#endif
#include "depth_addon_test_access.h"
#include <chrono>
#include <stdexcept>
#include "depth_selection_snapshot_test.h"

static void require_snapshot(bool value, const char *message)
{
	if (!value) throw std::runtime_error(message);
}

static auto snapshot_test_resources(unsigned count, unsigned clears)
{
	std::unordered_map<resource, depth_stencil_resource, resource_hash> resources;
	for (unsigned i = 0; i < count; ++i)
	{
		depth_stencil_resource info;
		info.desc.type = resource_type::texture_2d;
		info.desc.heap = memory_heap::default_;
		info.desc.usage = resource_usage::depth_stencil | resource_usage::copy_source;
		info.desc.texture = { i % 2 ? 1920u : 3840u, i % 2 ? 1080u : 2160u, 1, 1, format::d32_float, 1 };
		info.identity = i + 1;
		info.layout_epoch = i + 3;
		info.last_used_in_frame = 100 - i % 4;
		info.first_used_in_frame = i % 5;
		auto &frame = info.last_frame_stats;
		frame.total = { i * 300 + 6, i + 2, i % 4, i % 3,
			{ 0, 0, static_cast<float>(info.desc.texture.width), static_cast<float>(info.desc.texture.height), 0, 1 } };
		frame.current = frame.best_copy_stats = frame.total;
		frame.copied_viewport = { 16, 8, frame.total.last_viewport.width - 32, frame.total.last_viewport.height - 16, .1f, .9f };
		frame.copied_during_frame = i % 2;
		frame.copy_provenance_ambiguous = i % 3 == 0;
		if (i % 7 == 0) frame.total.last_viewport = {}; // Unknown draw/copy coverage.
		if (i % 5 == 0) frame.copied_viewport = {};
		frame.clears.resize(clears);
		for (auto &clear : frame.clears)
		{
			static_cast<draw_stats &>(clear) = frame.total;
			clear.copied_during_frame = true;
		}
		resources.emplace(resource { 0x1000 + uint64_t(i) * 16 }, std::move(info));
	}
	return resources;
}

static bool same_snapshot_viewport(const viewport &a, const viewport &b)
{
	return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height &&
		a.min_depth == b.min_depth && a.max_depth == b.max_depth;
}

static void check_snapshot_row(const depth_selection_resource &a, const depth_stencil_resource &b)
{
	require_snapshot(a.desc.type == b.desc.type && a.desc.heap == b.desc.heap && a.desc.usage == b.desc.usage &&
		a.desc.flags == b.desc.flags && a.desc.texture.width == b.desc.texture.width && a.desc.texture.height == b.desc.texture.height &&
		a.desc.texture.depth_or_layers == b.desc.texture.depth_or_layers && a.desc.texture.levels == b.desc.texture.levels &&
		a.desc.texture.format == b.desc.texture.format && a.desc.texture.samples == b.desc.texture.samples &&
		a.identity == b.identity && a.layout_epoch == b.layout_epoch && a.last_used_in_frame == b.last_used_in_frame &&
		a.first_used_in_frame == b.first_used_in_frame, "Snapshot changed resource identity, description or frame activity");
	const auto &x = a.last_frame_stats;
	const auto &y = b.last_frame_stats;
	require_snapshot(x.total.vertices == y.total.vertices && x.total.drawcalls == y.total.drawcalls &&
		x.total.drawcalls_indirect == y.total.drawcalls_indirect && x.total.scene_writes == y.total.scene_writes &&
		x.total.has_work() == y.total.has_work() && x.total.work_count() == y.total.work_count() &&
		same_snapshot_viewport(x.total.last_viewport, y.total.last_viewport) && same_snapshot_viewport(x.copied_viewport, y.copied_viewport) &&
		x.copied_during_frame == y.copied_during_frame && x.copy_provenance_ambiguous == y.copy_provenance_ambiguous,
		"Snapshot changed draw, mesh/copy work, viewport or copied-frame flags");
	require_snapshot(effective_capture_region(a) == effective_capture_region(b) &&
		candidate_matches_shape(a, 3840, 2160) == candidate_matches_shape(b, 3840, 2160),
		"Snapshot changed capture region or candidate shape admission");
}

static void check_depth_snapshot_contract()
{
	// UI work is an expiring request, independent of the capture-owner count.
	generic_depth_device_data demand_device;
	require_snapshot(!tracking_demand(demand_device).clear_history &&
		tracking_demand(demand_device).counters == s_frame_activity_trace, "Hidden API-native path retained candidate demand");
	request_depth_candidate_list(&demand_device);
	require_snapshot(tracking_demand(demand_device).counters && tracking_demand(demand_device).clear_history,
		"Opening manual depth selection did not request live candidate details");
	consume_depth_candidate_list_request(demand_device);
	require_snapshot(tracking_demand(demand_device).clear_history, "UI request lost its one-Present grace");
	consume_depth_candidate_list_request(demand_device);
	require_snapshot(!tracking_demand(demand_device).clear_history, "Closing the whole overlay left background history enabled");
	demand_device.generic_capture_users.store(1);
	require_snapshot(tracking_demand(demand_device).counters && !tracking_demand(demand_device).clear_history,
		"Generic/shared preservation lost numeric capture statistics with the UI hidden");

	// An already recorded command list may execute after UI/candidate demand
	// stops. Merge preserves transport evidence, never its old optional details.
	const resource tracked {0x1000};
	state_tracking cached(false), merged(false);
	auto &cached_stats = cached.stats_per_used_depth_stencil[tracked];
	cached_stats.total = {1200, 12, 3, 2, {0, 0, 1920, 1080, 0, 1}};
	cached_stats.current = cached_stats.best_copy_stats = cached_stats.total;
	cached_stats.clears.push_back({cached_stats.current, clear_op::fullscreen_draw, true});
	cached_stats.preservation_boundary = cached_stats.copied_during_frame = cached_stats.clear_zero = true;
	cached_stats.copied_viewport = cached_stats.total.last_viewport;
	cached_stats.capture_ticket.id = 77;
	cached_stats.capture_ticket.texture = 88;
	merged.merge(cached); // Existing detailed accumulation must be cleared too.
	merged.merge(cached, {false, false});
	const auto &paused = merged.stats_per_used_depth_stencil.at(tracked);
	require_snapshot(paused.total.has_work() && paused.current.has_work() && paused.total.work_count() == 0 &&
		paused.total.vertices == 0 && paused.total.drawcalls_indirect == 0 && paused.current.work_count() == 0 &&
		paused.best_copy_stats.vertices == 0 && paused.clears.empty(),
		"Cached command lists repopulated paused candidate counters/history");
	require_snapshot(paused.preservation_boundary && paused.copied_during_frame && paused.clear_zero &&
		paused.capture_ticket.id == 77 && paused.capture_ticket.texture == 88 &&
		same_snapshot_viewport(paused.total.last_viewport, cached_stats.total.last_viewport) &&
		same_snapshot_viewport(paused.copied_viewport, cached_stats.copied_viewport),
		"Pausing candidate details changed preservation eligibility, layout or capture ownership");
	state_tracking resumed(false);
	resumed.merge(merged, {true, false});
	require_snapshot(resumed.stats_per_used_depth_stencil.at(tracked).total.has_work() &&
		resumed.stats_per_used_depth_stencil.at(tracked).total.work_count() == 0,
		"Resuming counters invented workload for a recording observed while details were paused");
	resumed.merge(cached, {true, false});
	require_snapshot(resumed.stats_per_used_depth_stencil.at(tracked).total.vertices == 1200 &&
		resumed.stats_per_used_depth_stencil.at(tracked).total.drawcalls == 12 &&
		resumed.stats_per_used_depth_stencil.at(tracked).clears.empty(),
		"Resumed Generic/shared preservation changed known workload or retained hidden clear history");

	auto resources = snapshot_test_resources(16, 8);
	// Exercise the real capture-layout owner, including transfer/activity-only
	// frames and ambiguous captures. Aggregate rendering does not redefine a copy.
	auto layout = resources.begin()->second;
	layout.last_frame_stats.copied_during_frame = true;
	layout.last_frame_stats.copy_provenance_ambiguous = false;
	layout.last_frame_stats.copied_viewport = {16, 8, 1280, 720, 0, 1};
	observe_preserved_layout(layout, layout.last_frame_stats);
	const auto captured_region = effective_capture_region(layout);
	const auto captured_epoch = layout.layout_epoch;
	layout.last_frame_stats.copied_during_frame = false;
	layout.last_frame_stats.total.last_viewport = {0, 0, 100, 100, 0, 1};
	observe_preserved_layout(layout, layout.last_frame_stats);
	require_snapshot(effective_capture_region(layout) == captured_region && layout.layout_epoch == captured_epoch,
		"Activity-only viewport changed preserved capture layout");
	require_snapshot(effective_capture_region(depth_selection_resource(layout)) == captured_region,
		"Compact source snapshot lost authoritative captured crop");
	layout.last_frame_stats.copied_during_frame = true;
	layout.last_frame_stats.copy_provenance_ambiguous = true;
	layout.last_frame_stats.copied_viewport = {0, 0, 1024, 576, 0, 1};
	observe_preserved_layout(layout, layout.last_frame_stats);
	require_snapshot(effective_capture_region(layout) == captured_region, "Ambiguous copy changed numeric basis");
	layout.last_frame_stats.copy_provenance_ambiguous = false;
	observe_preserved_layout(layout, layout.last_frame_stats);
	require_snapshot(layout.layout_epoch == captured_epoch + 1 && effective_capture_region(layout).width == 1024,
		"A real changed captured viewport did not advance layout identity");
	resources.begin()->second.last_frame_stats.total = {};
	resources.begin()->second.last_used_in_frame = 90; // Live but inactive manual row must survive the snapshot.
	const auto legacy = resources; // Exact former on_begin_render_effects copy.
	const auto compact = snapshot_depth_selection(resources);
	require_snapshot(compact.size() == legacy.size(), "Snapshot lost depth resources");
	generic_depth_data old_data, new_data;
	auto old = legacy.begin();
	for (const auto &[resource, info] : compact)
	{
		require_snapshot(old != legacy.end() && old->first == resource, "Snapshot changed workload tie traversal order");
		check_snapshot_row(info, old->second);
		const capture_region retained { 32, 16, 1280, 720, info.layout_epoch };
		old_data.observed_regions[info.identity] = new_data.observed_regions[info.identity] = retained;
		observe_candidate(old_data, old->second, 3840, 2160);
		observe_candidate(new_data, info, 3840, 2160);
		require_snapshot(old_data.observed_regions.at(info.identity) == new_data.observed_regions.at(info.identity),
			"Snapshot changed retained/copy-ambiguous viewport evidence");
		old_data.observed_regions.erase(info.identity); new_data.observed_regions.erase(info.identity);
		observe_candidate(old_data, old->second, 3840, 2160);
		observe_candidate(new_data, info, 3840, 2160);
		require_snapshot(old_data.observed_regions.at(info.identity) == new_data.observed_regions.at(info.identity),
			"Snapshot changed viewport fallback without an earlier observed region");
		++old;
	}
	require_snapshot(old_data.selection.choose(100).id == new_data.selection.choose(100).id,
		"Snapshot changed candidate scoring");
	const auto resource = compact.front().first;
	const auto original = resources.at(resource);
	require_snapshot(find_depth_selection(compact, resource) != compact.end() && !compact.front().second.last_frame_stats.total.has_work(),
		"Snapshot filtered out a live inactive manual source");
	auto recropped = original;
	++recropped.layout_epoch;
	recropped.last_frame_stats.copied_during_frame = true;
	recropped.last_frame_stats.copy_provenance_ambiguous = false;
	recropped.last_frame_stats.copied_viewport = { 64, 32, 1280, 720, 0, 1 };
	old_data.native_calibrations[original.identity] = {}; new_data.native_calibrations[original.identity] = {};
	observe_candidate(old_data, recropped, 3840, 2160);
	observe_candidate(new_data, depth_selection_resource(recropped), 3840, 2160);
	require_snapshot(old_data.observed_regions.at(original.identity) == new_data.observed_regions.at(original.identity) &&
		new_data.observed_regions.at(original.identity).x == 64 && !old_data.native_calibrations.count(original.identity) &&
		!new_data.native_calibrations.count(original.identity), "Snapshot retained calibration across a changed content crop");
	resources.clear();
	for (const auto &[handle, info] : compact)
		check_snapshot_row(info, legacy.at(handle)); // No borrowed rows/history after unlock.
	auto replacement = original;
	replacement.identity += 1000;
	++replacement.layout_epoch;
	replacement.last_frame_stats.copied_during_frame = true;
	replacement.last_frame_stats.copy_provenance_ambiguous = false;
	replacement.last_frame_stats.copied_viewport = { 64, 32, 1280, 720, 0, 1 };
	resources.emplace(resource, replacement);
	const auto replaced = snapshot_depth_selection(resources);
	const auto fresh = find_depth_selection(replaced, resource);
	require_snapshot(fresh != replaced.end() && fresh->second.identity != compact.front().second.identity &&
		fresh->second.layout_epoch == replacement.layout_epoch && effective_capture_region(fresh->second).x == 64,
		"Native address reuse inherited old snapshot lifetime or crop");

	// Compare the removed live-ID pass + eligible-ID pass with the single final
	// pass. A manual inactive row is retained; a destroyed and a filtered row
	// are erased before delayed readbacks/selection can see their old evidence.
	const auto quality = sunshine_depth::depth_quality { sunshine_depth::content_kind::useful, 100 };
	generic_depth_data before, after;
	for (uint64_t id : { 1u, 2u, 3u, 9u })
	{
		depth_stencil_resource info = original;
		info.identity = id;
		info.last_frame_stats.total.drawcalls = id == 1 ? 0 : 10;
		info.last_frame_stats.total.scene_writes = 0;
		info.last_used_in_frame = id == 1 ? 90 : 100;
		observe_candidate(before, info, 3840, 2160);
		observe_candidate(after, depth_selection_resource(info), 3840, 2160);
		before.observed_ids.push_back(id); after.observed_ids.push_back(id);
		before.native_calibrations[id] = {}; after.native_calibrations[id] = {};
		for (uint64_t frame : { 98u, 99u, 100u })
		{
			before.selection.sample(id, quality, frame);
			after.selection.sample(id, quality, frame);
		}
	}
	const std::vector<uint64_t> live { 1, 2, 3 }, eligible { 1, 2 };
	for (const auto id : before.observed_ids)
		if (std::find(live.begin(), live.end(), id) == live.end())
		{
			before.selection.erase(id); before.observed_regions.erase(id); before.native_calibrations.erase(id);
		}
	prune_depth_selection(before, eligible);
	prune_depth_selection(after, eligible);
	for (uint64_t id : { 1u, 2u, 3u, 9u })
	{
		const bool retained = id <= 2;
		require_snapshot((after.selection.evidence(id) != nullptr) == retained &&
			after.observed_regions.count(id) == retained && after.native_calibrations.count(id) == retained &&
			before.observed_regions.count(id) == after.observed_regions.count(id) &&
			before.native_calibrations.count(id) == after.native_calibrations.count(id), "Single prune pass changed candidate lifetime");
		after.selection.sample(id, quality, 101);
		require_snapshot((after.selection.evidence(id) != nullptr) == retained, "Late readback resurrected a removed source");
	}
	require_snapshot(after.selection.evidence(1)->confirmed_good && after.native_calibrations.count(1),
		"Inactive manual row lost its retained histogram or calibration");
	observe_candidate(after, fresh->second, 3840, 2160);
	require_snapshot(!after.selection.evidence(replacement.identity)->confirmed_good,
		"Recycled resource inherited the prior lifetime's content evidence");
}

template <typename Snapshot>
static uint64_t depth_snapshot_checksum(const Snapshot &snapshot)
{
	uint64_t result = 0;
	for (const auto &[resource, info] : snapshot)
	{
		const auto &stats = info.last_frame_stats;
		result += resource.handle + info.identity * 17 + info.layout_epoch * 31 + info.last_used_in_frame + info.first_used_in_frame +
			info.desc.texture.width + info.desc.texture.height + stats.total.vertices + stats.total.work_count() +
			stats.total.drawcalls_indirect + static_cast<uint64_t>(stats.total.last_viewport.width + stats.total.last_viewport.height) +
			static_cast<uint64_t>(stats.copied_viewport.x + stats.copied_viewport.y + stats.copied_viewport.width + stats.copied_viewport.height) +
			stats.copied_during_frame + stats.copy_provenance_ambiguous;
	}
	return result;
}

extern "C" __declspec(dllexport) BOOL SunshineDepthTestSnapshot(unsigned buffers, unsigned clears, unsigned iterations,
	sunshine_depth_snapshot_test_result *output)
{
	if (!output) return FALSE;
	*output = {};
	try
	{
		require_snapshot(buffers <= 512 && clears <= 1024 && iterations && iterations <= 100000, "Invalid benchmark workload");
		check_depth_snapshot_contract();
		const auto resources = snapshot_test_resources(buffers, clears);
		output->clear_records = uint64_t(buffers) * clears;
		output->deep_value_bytes = buffers * sizeof(decltype(resources)::value_type) + output->clear_records * sizeof(clear_stats);
		output->compact_value_bytes = buffers * sizeof(depth_selection_snapshot::value_type);
		const uint64_t expected = depth_snapshot_checksum(resources) * iterations;
		const auto measure = [&](bool deep) {
			uint64_t checksum = 0;
			const auto start = std::chrono::steady_clock::now();
			for (unsigned i = 0; i < iterations; ++i)
			{
				if (deep) { const auto snapshot = resources; checksum += depth_snapshot_checksum(snapshot); }
				else { const auto snapshot = snapshot_depth_selection(resources); checksum += depth_snapshot_checksum(snapshot); }
			}
			const auto stop = std::chrono::steady_clock::now();
			require_snapshot(checksum == expected, "Timed snapshots did not produce the same consumed values");
			output->checksum ^= checksum;
			return std::chrono::duration<double, std::nano>(stop - start).count() / iterations;
		};
		measure(true); measure(false); // Warm both allocation paths outside reported rounds.
		for (unsigned round = 0; round < output->rounds; ++round)
		{
			if (round % 2) { output->compact_ns[round] = measure(false); output->deep_ns[round] = measure(true); }
			else { output->deep_ns[round] = measure(true); output->compact_ns[round] = measure(false); }
		}
		output->checksum = expected;
		return TRUE;
	}
	catch (const std::exception &error)
	{
		std::snprintf(output->error, sizeof(output->error), "%s", error.what());
		return FALSE;
	}
}

// Test adapter uses the same lifetime lookup and override handover as the UI.
// It is absent from the production DLL and never injects depth pixels.
extern "C" __declspec(dllexport) BOOL SunshineDepthTestProviderStatus(effect_runtime *runtime,
	sunshine_streamline::provider::source_status *output)
{
	if (!s_registered || runtime == nullptr || output == nullptr) return FALSE;
	*output = describe_streamline_source(runtime);
	return TRUE;
}

extern "C" __declspec(dllexport) BOOL SunshineDepthTestSelectManual(effect_runtime *runtime, uint64_t native_resource)
{
	if (!s_registered || runtime == nullptr)
		return FALSE;
	auto *data = runtime->get_private_data<generic_depth_data>();
	auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (data == nullptr || device_data == nullptr)
		return FALSE;
	if (native_resource == 0)
	{
		set_manual_depth_override(*data, {}, 0);
		s_sunshine_auto = true;
		reshade::set_config_value(nullptr, "SUNSHINE_DEPTH", "AutoSelectSceneDepth", s_sunshine_auto);
		return TRUE;
	}
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		const auto found = device_data->depth_stencil_resources.find(resource { native_resource });
		if (found == device_data->depth_stencil_resources.end() || found->second.last_used_in_frame != device_data->frame_index)
			return FALSE;
		set_manual_depth_override(*data, found->first, found->second.identity);
	}
	return TRUE;
}

extern "C" __declspec(dllexport) BOOL SunshineDepthTestManualState(effect_runtime *runtime, uint64_t *native_resource, BOOL *recovering)
{
	if (!s_registered || runtime == nullptr || native_resource == nullptr || recovering == nullptr)
		return FALSE;
	const auto *data = runtime->get_private_data<generic_depth_data>();
	if (data == nullptr)
		return FALSE;
	*native_resource = data->override_depth_stencil.handle;
	*recovering = data->manual_recovery_probing ? TRUE : FALSE;
	return TRUE;
}

// Calls the real event handlers on an isolated snapshot of this runtime's
// command state. It checks accounting only; no mesh/copy GPU command is issued.
extern "C" __declspec(dllexport) BOOL SunshineDepthTestActivityAccounting(effect_runtime *runtime, uint64_t native_resource)
{
	if (!s_registered || runtime == nullptr || native_resource == 0 || s_streamline_probe_events) return FALSE;
	auto *queue = runtime->get_command_queue();
	if (queue == nullptr) return FALSE;
	auto *cmd = queue->get_immediate_command_list();
	// ReShade's internal D3D12 list does not raise application init_command_list.
	// Own scratch tracking only for this test; issue no commands and remove it on exit.
	const bool temporary_tracking = cmd->get_private_data<state_tracking>() == nullptr;
	if (temporary_tracking) cmd->create_private_data<state_tracking>(false);
	struct release_tracking {
		command_list *cmd;
		bool owned;
		~release_tracking() { if (owned) cmd->destroy_private_data<state_tracking>(); }
	} tracking {cmd, temporary_tracking};
	auto *ptr = cmd->get_private_data<state_tracking>();
	if (ptr == nullptr) return FALSE;
	auto &state = *ptr;
	struct restore_state {
		state_tracking &state;
		decltype(state.stats_per_used_depth_stencil) stats;
		resource depth;
		uint64_t lifetime;
		viewport view;
		bool first;
		~restore_state() {
			const std::unique_lock<std::shared_mutex> lock(s_mutex);
			state.stats_per_used_depth_stencil = std::move(stats);
			state.current_depth_stencil = depth; state.current_depth_lifetime = lifetime;
			state.current_viewport = view; state.first_draw_since_bind = first;
		}
	};
	std::unique_lock<std::shared_mutex> lock(s_mutex);
	restore_state saved {state, std::move(state.stats_per_used_depth_stencil), state.current_depth_stencil,
		state.current_depth_lifetime, state.current_viewport, state.first_draw_since_bind};
	state.stats_per_used_depth_stencil.clear();
	state.current_depth_stencil = {native_resource}; state.current_depth_lifetime = 0;
	state.current_viewport = {0, 0, 1280, 720, 0, 1}; state.first_draw_since_bind = true;
	lock.unlock();
	const resource target {native_resource};
	bool ok = true;
	const auto checkpoint = [&](const char *label) {
		if (!ok) sunshine_log::message(reshade::log::level::error, label);
		return ok;
	};
	for (const auto type : {indirect_command::dispatch, indirect_command::dispatch_rays})
		ok &= !on_draw_indirect(cmd, type, {}, 0, 3, 0);
	for (const auto type : {indirect_command::unknown, indirect_command::draw, indirect_command::draw_indexed, indirect_command::dispatch_mesh})
		ok &= !on_draw_indirect(cmd, type, {}, 0, 0, 0);
	ok &= !on_dispatch_mesh(cmd, 0, 1, 1) && !on_dispatch_mesh(cmd, 1, 0, 1) && !on_dispatch_mesh(cmd, 1, 1, 0);
	ok &= !on_draw(cmd, 0, 1, 0, 0) && !on_draw_indexed(cmd, 3, 0, 0, 0, 0);
	ok &= state.stats_per_used_depth_stencil.empty() && state.first_draw_since_bind;
	if (!checkpoint("Depth activity test: no-op accounting failed")) return FALSE;
	ok &= !on_dispatch_mesh(cmd, 2, 3, 4);
	auto &mesh = state.stats_per_used_depth_stencil[target];
	ok &= mesh.total.has_work() && mesh.total.scene_writes == 1 && mesh.current.scene_writes == 1 &&
		mesh.total.drawcalls == 0 && mesh.total.vertices == 0 && mesh.current.vertices == 0 &&
		mesh.current.last_viewport.width == 1280 && !state.first_draw_since_bind;
	state_tracking merged(false); merged.merge(state);
	ok &= merged.stats_per_used_depth_stencil[target].total.scene_writes == 1 &&
		merged.stats_per_used_depth_stencil[target].current.scene_writes == 1;
	merged.reset(); ok &= merged.stats_per_used_depth_stencil.empty();
	if (!checkpoint("Depth activity test: mesh/merge/reset accounting failed")) return FALSE;
	state.stats_per_used_depth_stencil.clear();
	ok &= !on_draw(cmd, 3, 2, 0, 0) && !on_draw_indexed(cmd, 6, 1, 0, 0, 0);
	for (const auto type : {indirect_command::unknown, indirect_command::draw, indirect_command::draw_indexed, indirect_command::dispatch_mesh})
		ok &= !on_draw_indirect(cmd, type, {}, 0, 2, 0);
	const auto &drawn = state.stats_per_used_depth_stencil[target];
	ok &= drawn.total.drawcalls == 10 && drawn.total.drawcalls_indirect == 8 && drawn.total.vertices == 12 && drawn.total.scene_writes == 0;
	if (!checkpoint("Depth activity test: raster/indirect accounting failed")) return FALSE;
	state.stats_per_used_depth_stencil.clear();
	for (const subresource_box box : {subresource_box{0, 0, 0, 0, 1, 1}, subresource_box{0, 0, 0, 1, 0, 1}, subresource_box{0, 0, 0, 1, 1, 0}}) {
		ok &= !on_copy_texture_region(cmd, {}, 0, &box, target, 0, nullptr, filter_mode::min_mag_mip_point);
		ok &= !on_copy_texture_region(cmd, {}, 0, nullptr, target, 0, &box, filter_mode::min_mag_mip_point);
		ok &= !on_copy_buffer_to_texture(cmd, {}, 0, 0, 0, target, 0, &box);
		ok &= !on_resolve_texture_region(cmd, {}, 0, &box, target, 0, 0, 0, 0, format::unknown);
	}
	ok &= state.stats_per_used_depth_stencil.empty(); // Explicit zero-volume transfers cannot refresh clear-only depth.
	ok &= !on_copy_texture_region(cmd, {}, 0, nullptr, target, 1, nullptr, filter_mode::min_mag_mip_point);
	ok &= state.stats_per_used_depth_stencil.empty(); // A stencil plane or other mip is not the sampled depth.
	if (!checkpoint("Depth activity test: empty/subresource copy accounting failed")) return FALSE;
	ok &= !on_copy_resource(cmd, {}, target);
	ok &= !on_copy_texture_region(cmd, {}, 0, nullptr, target, 0, nullptr, filter_mode::min_mag_mip_point);
	ok &= !on_copy_buffer_to_texture(cmd, {}, 0, 0, 0, target, 0, nullptr);
	ok &= !on_resolve_texture_region(cmd, {}, 0, nullptr, target, 0, 0, 0, 0, format::unknown);
	const auto &copied = state.stats_per_used_depth_stencil[target];
	ok &= copied.total.scene_writes == 4 && copied.current.scene_writes == 4 && copied.total.has_work() &&
		copied.total.drawcalls == 0 && copied.total.vertices == 0 && copied.total.last_viewport.width == 0;
	if (!checkpoint("Depth activity test: base-depth copy accounting failed")) return FALSE;
	if (!s_frame_activity_trace)
	{
		auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
		if (device_data == nullptr) return FALSE;
		struct restore_demand {
			generic_depth_device_data &data;
			unsigned captures, list;
			~restore_demand() {
				data.generic_capture_users.store(captures);
				data.candidate_list_request.store(list);
			}
		} demand {*device_data, device_data->generic_capture_users.exchange(0), device_data->candidate_list_request.exchange(0)};
		state.stats_per_used_depth_stencil.clear();
		ok &= !on_draw(cmd, 3, 1, 0, 0) && !on_draw_indirect(cmd, indirect_command::draw, {}, 0, 2, 0) &&
			!on_dispatch_mesh(cmd, 1, 1, 1) && !on_copy_resource(cmd, {}, target);
		auto &passive = state.stats_per_used_depth_stencil[target];
		ok &= passive.total.has_work() && passive.current.has_work() && passive.total.work_count() == 0 &&
			passive.total.vertices == 0 && passive.total.drawcalls_indirect == 0 && passive.total.last_viewport.width == 1280;
		// Capture is disabled here, so this exercises the real boundary classifier
		// without recording a GPU copy or changing any existing backup assignment.
		on_clear_depth_impl(cmd, state, target, clear_op::unbind_depth_stencil_view);
		ok &= passive.preservation_boundary && passive.clears.empty();
		request_depth_candidate_list(device_data);
		ok &= !on_draw(cmd, 3, 1, 0, 0) && passive.total.drawcalls == 1 && passive.total.vertices == 3;
		consume_depth_candidate_list_request(*device_data);
		consume_depth_candidate_list_request(*device_data);
		ok &= !on_draw(cmd, 3, 1, 0, 0) && passive.total.has_work() && passive.total.work_count() == 0;
		device_data->generic_capture_users.store(1);
		ok &= !on_draw(cmd, 3, 1, 0, 0) && passive.total.drawcalls == 1 && passive.total.vertices == 3;
		if (!checkpoint("Depth activity test: hidden API counters/UI expiry/preservation recovery failed")) return FALSE;
	}
	state.stats_per_used_depth_stencil.clear(); state.current_depth_stencil = {};
	ok &= !on_dispatch_mesh(cmd, 1, 1, 1) && state.stats_per_used_depth_stencil.empty();
	return ok ? TRUE : FALSE;
}

extern "C" __declspec(dllexport) BOOL SunshineDepthTestCaptureDemand(effect_runtime *runtime,
	BOOL *enabled, unsigned *users, uint64_t *transitions)
{
	if (!runtime || !enabled || !users || !transitions) return FALSE;
	const auto *data = runtime->get_private_data<generic_depth_data>();
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (!data || !device_data) return FALSE;
	*enabled = data->generic_capture_enabled ? TRUE : FALSE;
	*users = device_data->generic_capture_users.load(std::memory_order_relaxed);
	*transitions = data->generic_capture_transitions;
	return TRUE;
}

extern "C" __declspec(dllexport) BOOL SunshineDepthTestFrame(effect_runtime *runtime, sunshine_depth::frame_depth *output)
{
	return output != nullptr && sunshine_depth::get_frame_depth(runtime, *output, sunshine_depth::depth_orientation::automatic, false);
}

// The actual preserved copy behind the selected D3D12 backup published for this
// Present: the before-clear copy into the capture owner's ticket texture, in the
// game's recording. The frame's depth_copy is its runtime-list forward instead.
extern "C" __declspec(dllexport) BOOL SunshineDepthTestPreservedCopy(effect_runtime *runtime,
	sunshine_streamline::content::copy_snapshot *copy, uint64_t *texture, uint64_t *lifetime)
{
	if (!s_registered || runtime == nullptr || copy == nullptr || texture == nullptr || lifetime == nullptr) return FALSE;
	*copy = {}; *texture = *lifetime = 0;
	if (runtime->get_device()->get_api() != device_api::d3d12) return FALSE;
	const auto *data = runtime->get_private_data<generic_depth_data>();
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (!data || !device_data || !data->native_access_open || !data->capture_ready || data->selected_record.direct ||
		data->native_access_present != device_data->native_present_index ||
		!validate_capture_record(runtime, *data, *device_data, data->selected_record)) return FALSE;
	const resource source{data->selected_record.source};
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	const auto found = device_data->depth_stencil_resources.find(source);
	const auto backup = std::find_if(device_data->depth_stencil_backups.begin(), device_data->depth_stencil_backups.end(),
		[source](const auto &entry) { return entry.depth_stencil == source; });
	if (found == device_data->depth_stencil_resources.end() || found->second.identity != data->selected_record.identity ||
		backup == device_data->depth_stencil_backups.end()) return FALSE;
	const auto &ticket = found->second.last_frame_stats.capture_ticket;
	if (!ticket || backup->published_ticket != ticket.id || backup->published_present != device_data->native_present_index) return FALSE;
	*copy = found->second.last_frame_stats.depth_copy;
	*texture = ticket.texture; *lifetime = ticket.resource_id;
	return TRUE;
}

extern "C" __declspec(dllexport) BOOL SunshineDepthTestLegacyCalibrationState(effect_runtime *runtime, unsigned *requested, unsigned *entries)
{
	if (!s_registered || runtime == nullptr || requested == nullptr || entries == nullptr)
		return FALSE;
	const auto *data = runtime->get_private_data<generic_depth_data>();
	if (data == nullptr)
		return FALSE;
	*requested = data->native_depth_requested ? 1u : 0u;
	*entries = static_cast<unsigned>(data->native_calibrations.size());
	return TRUE;
}

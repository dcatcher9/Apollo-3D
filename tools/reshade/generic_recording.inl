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

// Generic depth's command recording, a part of depth_addon.cpp (included at
// its end, in the same translation unit): the ReShade draw, binding, clear,
// render pass, transfer and execute events that fill state_tracking
// (generic_recording.h), and resolve_transfer_lineage, which on_present runs
// on the queue's merged recording. Declared in depth_addon.cpp.
#pragma once

SUNSHINE_DEPTH_ADDON_INTERNAL bool on_draw(command_list *cmd_list, uint32_t vertices, uint32_t instances, uint32_t, uint32_t)
{
	if (!vertices || !instances) return false;
	auto &state = *cmd_list->get_private_data<state_tracking>();
	if (state.current_depth_stencil == 0)
		return false; // This is a draw call with no depth-stencil bound

	// If this is queue state (happens if this is a immediate command list), need to protect access to it, since another thread may be in a present call, which can reset it
	std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
	if (state.is_queue)
		lock.lock();

	depth_stencil_frame_stats &stats = state.stats_per_used_depth_stencil[state.current_depth_stencil];
	const auto demand = tracking_demand(cmd_list->get_device());
	stats.limit_tracking(demand);
	stats.total.observed_work = true;
	stats.observe_draw(state.position());
	if (demand.counters)
	{
		stats.total.vertices += vertices * instances;
		stats.total.drawcalls += 1;
	}
	stats.observe_viewport(state.current_viewport);

	return false;
}
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_draw_indexed(command_list *cmd_list, uint32_t indices, uint32_t instances, uint32_t, int32_t, uint32_t)
{
	return on_draw(cmd_list, indices, instances, 0, 0);
}
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_draw_indirect(command_list *cmd_list, indirect_command type, resource, uint64_t, uint32_t draw_count, uint32_t)
{
	if (!draw_count || type == indirect_command::dispatch || type == indirect_command::dispatch_rays)
		return false; // Compute/ray dispatches do not rasterize into the bound DSV.
	// ReShade 6.8 reports D3D12 ExecuteIndirect as 'unknown' even for raster draws.
	// Preserve that existing activity evidence until the backend exposes its signature.

	auto &state = *cmd_list->get_private_data<state_tracking>();
	if (state.current_depth_stencil == 0)
		return false; // This is a draw call with no depth-stencil bound

	// If this is queue state (happens if this is a immediate command list), need to protect access to it, since another thread may be in a present call, which can reset it
	std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
	if (state.is_queue)
		lock.lock();

	depth_stencil_frame_stats &stats = state.stats_per_used_depth_stencil[state.current_depth_stencil];
	const auto demand = tracking_demand(cmd_list->get_device());
	stats.limit_tracking(demand);
	stats.total.observed_work = true;
	stats.observe_draw(state.position());
	if (demand.counters)
	{
		stats.total.drawcalls += draw_count;
		stats.total.drawcalls_indirect += draw_count;
	}
	stats.observe_viewport(state.current_viewport);

	return false;
}

static void on_bind_viewport(command_list *cmd_list, uint32_t first, uint32_t count, const viewport *viewport)
{
	if (first != 0 || count == 0)
		return; // Only interested in the main viewport

	auto &state = *cmd_list->get_private_data<state_tracking>();
	state.current_viewport = viewport[0];
}
// No binding is a copy point (Generic depth copies at the end of the frame),
// so the binding kind, an actual OMSetRenderTargets or a render pass's
// synthesized one, is unused.
SUNSHINE_DEPTH_ADDON_INTERNAL void on_bind_depth_stencil_impl(command_list *cmd_list, resource_view depth_stencil_view, bool)
{
	auto &state = *cmd_list->get_private_data<state_tracking>();

	const resource depth_stencil = (depth_stencil_view != 0) ? cmd_list->get_device()->get_resource_from_view(depth_stencil_view) : resource { 0 };

	state.current_depth_stencil = depth_stencil;
}
static void on_bind_depth_stencil(command_list *cmd_list, uint32_t, const resource_view *, resource_view depth_stencil_view)
{
	on_bind_depth_stencil_impl(cmd_list, depth_stencil_view, true);
}
static bool on_clear_depth_stencil(command_list *cmd_list, resource_view dsv, const float *depth, const uint8_t *, uint32_t rect_count, const rect *)
{
	// Ignore clears that do not affect the depth buffer (stencil clears)
	if (dsv != 0 && depth != nullptr)
	{
		auto &state = *cmd_list->get_private_data<state_tracking>();

		const resource depth_stencil = cmd_list->get_device()->get_resource_from_view(dsv);
		const auto view = cmd_list->get_device()->get_resource_view_desc(dsv);
		const bool sampled_plane = view.texture.first_level == 0 && view.texture.first_layer == 0;

		// The clear needs DEPTH_WRITE (the D3D12 clear contract): it proves the
		// state of subresource 0 when its list executes (proven_state.h).
		if (cmd_list->get_device()->get_api() == device_api::d3d12 && sampled_plane)
			capture_inputs::observe_depth_clear(cmd_list, depth_stencil);
		// The game clear changes the source; no copy is made here.

		std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
		if (state.is_queue)
			lock.lock();
		auto &stats = state.stats_per_used_depth_stencil[depth_stencil];
		stats.limit_tracking(tracking_demand(cmd_list->get_device()));
		stats.clear_zero |= *depth == 0.0f;
		stats.clear_one |= *depth == 1.0f;
		stats.clear_other |= *depth != 0.0f && *depth != 1.0f;
		stats.depth_erased |= sampled_plane;
		// A full clear of the sampled plane erases the extent drawn so far; the
		// next draw starts the capture crop again (observe_viewport). Only the
		// D3D10/11/12 clears are known to cover the whole view.
		stats.observe_depth_clear(cmd_list->get_device()->get_api(), rect_count, sampled_plane, state.position());
	}

	return false;
}
static bool on_begin_render_pass_with_depth_stencil(command_list *cmd_list, uint32_t, const render_pass_render_target_desc *, const render_pass_depth_stencil_desc *depth_stencil_desc, render_pass_flags)
{
	if (depth_stencil_desc != nullptr && depth_stencil_desc->depth_load_op == render_pass_load_op::clear)
		on_clear_depth_stencil(cmd_list, depth_stencil_desc->view, &depth_stencil_desc->clear_depth, nullptr, 0, nullptr);
	else if (depth_stencil_desc != nullptr && depth_stencil_desc->view != 0 && depth_stencil_desc->depth_load_op == render_pass_load_op::discard &&
		cmd_list->get_device()->get_api() == device_api::d3d12)
	{
		// A D3D12 depth DISCARD (observe_depth_discard). Vulkan's DONT_CARE is
		// left to the draw-statistics picker as before.
		device *const device = cmd_list->get_device();
		const auto view = device->get_resource_view_desc(depth_stencil_desc->view);
		if (view.texture.first_level == 0 && view.texture.first_layer == 0)
		{
			auto &state = *cmd_list->get_private_data<state_tracking>();
			const resource depth_stencil = device->get_resource_from_view(depth_stencil_desc->view);
			std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
			if (state.is_queue)
				lock.lock();
			auto &stats = state.stats_per_used_depth_stencil[depth_stencil];
			stats.limit_tracking(tracking_demand(device));
			stats.observe_depth_discard(state.position());
		}
	}
	// ReShade runs this event before the native BeginRenderPass, so the clear
	// above is observed outside the pass. The capture owner sees the pass from
	// here on, before the synthesized binding below.
	capture_inputs::list_event(cmd_list, sunshine_streamline::depth_capture::list_event::pass_begin);

	// If render pass has depth store operation set to 'discard', any copy performed after the render pass will likely contain broken data, so can only hope that the depth buffer can be copied before that ...

	on_bind_depth_stencil_impl(cmd_list, depth_stencil_desc != nullptr ? depth_stencil_desc->view : resource_view {}, false);
	return false;
}

static bool on_end_render_pass(command_list *cmd_list)
{
	capture_inputs::list_event(cmd_list, sunshine_streamline::depth_capture::list_event::pass_end);
	auto &state = *cmd_list->get_private_data<state_tracking>();
	state.render_pass_depth = {};
	return false;
}
// Call under the queue-state lock when needed. These writes establish activity,
// not useful contents or a vertex estimate; the existing histogram still qualifies them.
static depth_stencil_frame_stats &record_scene_write(state_tracking &state, resource target, const viewport &view, depth_tracking_demand demand, bool mesh = false)
{
	auto &stats = state.stats_per_used_depth_stencil[target];
	stats.limit_tracking(demand);
	stats.total.observed_work = true;
	if (mesh) stats.observe_draw(state.position());
	if (demand.counters)
	{
		++stats.total.scene_writes;
		stats.total.mesh_dispatches += mesh;
	}
	stats.observe_viewport(view);
	return stats;
}

// full: the transfer replaces the whole sampled plane (no destination box or
// offset, and no source box or, from a tracked depth-stencil, one spanning
// its whole plane: whole_plane_box).
static void record_copied_depth(command_list *cmd_list, resource source, resource dest, uint32_t subresource, bool full,
	const subresource_box *source_box = nullptr)
{
	if (dest == 0 || subresource != 0) return; // The shader samples the first depth plane/level/layer.
	auto *device_data = cmd_list->get_device()->get_private_data<generic_depth_device_data>();
	if (device_data == nullptr) return;
	auto &state = *cmd_list->get_private_data<state_tracking>();
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	const auto found = device_data->depth_stencil_resources.find(dest);
	if (found == device_data->depth_stencil_resources.end()) return; // Skip color and our sampling backups.
	if (full && source_box)
	{
		const auto from = device_data->depth_stencil_resources.find(source);
		full = from != device_data->depth_stencil_resources.end() && whole_plane_box(source_box, from->second.desc);
	}
	// A transfer is activity, not proof of a full raster viewport. In particular,
	// ReShade can report a null destination box for a partial origin copy.
	// Its source and position let on_present tell a copy of another candidate
	// (resolve_transfer_lineage) from depth that exists nowhere else.
	record_scene_write(state, dest, {}, tracking_demand(*device_data)).observe_transfer(source, state.position(), full);
}

// A destination whose only work this frame was a copy holds no depth of its
// own. When the copy's source is a same-geometry candidate on the same queue,
// the source holds the same or newer depth at the end of the frame: the copy
// was made before the source drew this frame (a history copy, older depth), or
// it duplicates depth the source still holds. Such a transfer stops counting
// as activity, so the destination is neither captured as current depth nor
// selected; the source is. With no clear of its own either, the destination
// is left out of the frame, as if untouched: its allocation, its evidence and
// its activity grace stay as its last own work left them, so a member of an
// alternating set that receives a copy between its turns stays active. The
// one exception is a whole copy of depth the source drew and then overwrote
// (last_overwrite: a clear of its sampled plane, with or without rectangles,
// or a render pass that clears or DISCARDs it, after the copy): only the
// destination may still hold that pass, so it stays a candidate, judged by its
// content, and takes the source's clear convention for its depth direction. A
// source overwritten only by drawing is not seen as such. A copy from anything
// else (a buffer, an MSAA or mipped depth, another size or format, or a source
// that is not a tracked depth-stencil) keeps today's activity, as does a
// destination that also drew. A same-geometry candidate whose work this frame
// ran on another queue has none in this queue's recording, no position to
// order the copy against, so the copy is treated as above: never a lost pass,
// never current depth of its destination (fail-safe; a pass that queue drew
// and then cleared is not kept; check_transfer_lineage's no-work case).
// Generic content
// selection only (D3D11 and D3D12); the other APIs' draw-statistics picker
// keeps ReShade's accounting.
SUNSHINE_DEPTH_ADDON_INTERNAL void resolve_transfer_lineage(state_tracking &state, const generic_depth_device_data &device_data)
{
	std::array<resource, 8> untouched {};
	size_t untouched_count = 0;
	for (auto &[dest, stats] : state.stats_per_used_depth_stencil)
	{
		if (!stats.transfer_position || stats.first_draw) continue;
		const auto target = device_data.depth_stencil_resources.find(dest);
		const auto source = device_data.depth_stencil_resources.find(stats.transfer_source);
		if (target == device_data.depth_stencil_resources.end() || source == device_data.depth_stencil_resources.end() ||
			source == target || !content_supported(source->second.desc) ||
			source->second.desc.texture.width != target->second.desc.texture.width ||
			source->second.desc.texture.height != target->second.desc.texture.height ||
			source->second.desc.texture.format != target->second.desc.texture.format) continue;
		const auto found = state.stats_per_used_depth_stencil.find(stats.transfer_source);
		const depth_stencil_frame_stats *producer = found == state.stats_per_used_depth_stencil.end() ? nullptr : &found->second;
		if (stats.transfer_full && producer && producer->first_draw && producer->first_draw < stats.transfer_position &&
			producer->last_overwrite > stats.transfer_position)
		{
			stats.clear_zero |= producer->clear_zero;
			stats.clear_one |= producer->clear_one;
			stats.clear_other |= producer->clear_other;
			continue;
		}
		stats.total.observed_work = false;
		stats.total.scene_writes = 0;
		stats.transfer_alias = true;
		if (!stats.clear_zero && !stats.clear_one && !stats.clear_other && untouched_count != untouched.size())
			untouched[untouched_count++] = dest;
	}
	// More than the bounded list stay as clear-only frames: inactive, never current.
	for (size_t i = 0; i != untouched_count; ++i)
		state.stats_per_used_depth_stencil.erase(untouched[i]);
}

SUNSHINE_DEPTH_ADDON_INTERNAL bool on_dispatch_mesh(command_list *cmd_list, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
	auto &state = *cmd_list->get_private_data<state_tracking>();
	if (!groups_x || !groups_y || !groups_z || state.current_depth_stencil == 0) return false;
	std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
	if (state.is_queue) lock.lock();
	record_scene_write(state, state.current_depth_stencil, state.current_viewport, tracking_demand(cmd_list->get_device()), true);
	return false;
}
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_copy_resource(command_list *cmd_list, resource source, resource dest)
{
	record_copied_depth(cmd_list, source, dest, 0, true);
	return false;
}
static bool empty_copy_box(const subresource_box *box)
{
	return box != nullptr && (box->left >= box->right || box->top >= box->bottom || box->front >= box->back);
}
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_copy_texture_region(command_list *cmd_list, resource source, uint32_t source_subresource,
	const subresource_box *source_box, resource dest, uint32_t subresource, const subresource_box *dest_box, filter_mode)
{
	if (empty_copy_box(source_box) || empty_copy_box(dest_box)) return false;
	record_copied_depth(cmd_list, source, dest, subresource, source_subresource == 0 && !dest_box, source_box);
	return false;
}
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_copy_buffer_to_texture(command_list *cmd_list, resource, uint64_t, uint32_t, uint32_t,
	resource dest, uint32_t subresource, const subresource_box *dest_box)
{
	if (empty_copy_box(dest_box)) return false;
	record_copied_depth(cmd_list, {}, dest, subresource, false); // A buffer is never another depth candidate.
	return false;
}
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_resolve_texture_region(command_list *cmd_list, resource source, uint32_t source_subresource,
	const subresource_box *source_box, resource dest, uint32_t subresource, uint32_t dest_x, uint32_t dest_y, uint32_t dest_z, format)
{
	if (empty_copy_box(source_box)) return false;
	record_copied_depth(cmd_list, source, dest, subresource, source_subresource == 0 && !source_box && !dest_x && !dest_y && !dest_z);
	return false;
}
// Its only work was the StreamlineCameraProbe content ledger's; it stays
// registered so the registered event set is unchanged.
static void on_content_barrier(command_list *, uint32_t, const resource *, const resource_usage *, const resource_usage *)
{
}

static void on_reset(command_list *cmd_list)
{
	capture_inputs::list_event(cmd_list, sunshine_streamline::depth_capture::list_event::reset);
	auto &target_state = *cmd_list->get_private_data<state_tracking>();
	target_state.reset();
}
static void on_close(command_list *cmd_list)
{
	capture_inputs::list_event(cmd_list, sunshine_streamline::depth_capture::list_event::closed);
}
static void on_execute_primary(command_queue *queue, command_list *cmd_list)
{
	// Skip merging state when this execution event is just the immediate command list getting flushed
	if (cmd_list == queue->get_immediate_command_list())
		return;

	auto &target_state = *queue->get_private_data<state_tracking>();
	const auto &source_state = *cmd_list->get_private_data<state_tracking>();
	assert(target_state.is_queue && !source_state.is_queue);

	// Need to protect access to the queue state, since another thread may be in a present call, which can reset this state
	const std::unique_lock<std::shared_mutex> lock(s_mutex);

	target_state.merge(source_state, tracking_demand(queue->get_device()));
}
static void on_execute_secondary(command_list *cmd_list, command_list *secondary_cmd_list)
{
	auto &target_state = *cmd_list->get_private_data<state_tracking>();
	const auto &source_state = *secondary_cmd_list->get_private_data<state_tracking>();

	// If this is a secondary command list that was recorded without a depth-stencil binding, but is now executed using a depth-stencil binding, handle it as if an indirect draw call was performed to ensure the depth-stencil is tracked
	if (target_state.current_depth_stencil != 0 && source_state.current_depth_stencil == 0 && source_state.stats_per_used_depth_stencil.empty())
	{
		target_state.current_viewport = source_state.current_viewport;

		on_draw_indirect(cmd_list, indirect_command::draw, { 0 }, 0, 1, 0);
	}
	else
	{
		// If this is queue state (happens if this is a immediate command list), need to protect access to it, since another thread may be in a present call, which can reset it
		std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
		if (target_state.is_queue)
			lock.lock();

		target_state.merge(source_state, tracking_demand(cmd_list->get_device()));
	}
}

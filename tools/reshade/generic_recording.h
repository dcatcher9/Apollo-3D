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

// Generic depth's per-command-list recording tracker: what each command list
// or queue recorded into each depth-stencil this frame (state_tracking and
// its depth_stencil_frame_stats). generic_depth_state.h includes it at this
// position, after draw_stats and before the per-device data that keeps a
// resource's last frame statistics. The recording events that fill it are in
// generic_recording.inl.
#pragma once

#ifndef SUNSHINE_DEPTH_ADDON_SHARED
#error "Include depth_addon_test_access.h, which defines the linkage of this state"
#endif

// A copy's source box spans the whole sampled plane (mip 0, layer 0) of a
// texture described by desc; no box is the whole subresource. On D3D11 and
// D3D12 a depth-stencil copies only whole subresources, so such a box is the
// one legal box there, and the copy replaces the plane as a box-less one does.
SUNSHINE_DEPTH_ADDON_SHARED bool whole_plane_box(const subresource_box *box, const resource_desc &desc)
{
	return box == nullptr || (box->left == 0 && box->top == 0 && box->front == 0 &&
		box->right >= desc.texture.width && box->bottom >= desc.texture.height && box->back >= 1);
}

struct depth_stencil_frame_stats
{
	draw_stats total;
	viewport copied_viewport = {}; // Last submitted copy, independent of greatest workload.
	bool copy_provenance_ambiguous = false;
	bool copied_during_frame = false;
	// Pixel storage and GPU lifetime belong to the common capture owner. This
	// lease survives recording reset until the submitted frame is consumed.
	sunshine_streamline::depth_capture::preservation_ticket capture_ticket;
	// Only this reservation may publish an EOF capture for the source/frame.
	uint64_t preservation_reservation = 0;
	bool clear_zero = false, clear_one = false, clear_other = false;
	// This frame cleared the sampled plane's depth (with or without rectangles,
	// and a render pass's depth CLEAR) or began a D3D12 render pass that DISCARDs it:
	// what was there is gone. The hold's cleared gate (judge_hold_gate) reads it.
	bool depth_erased = false;
	// This frame's copy is the end-of-frame one (preserve_depth_at_effects).
	// Counted for the status line only.
	bool end_of_frame_copy = false;
	// total.last_viewport is the capture crop: the largest viewport drawn since
	// the last full depth clear. A clear followed by more drawing erased the
	// earlier extent, so a larger pass the game cleared away cannot widen the
	// scene's crop; a trailing clear with no draw after it keeps the extent.
	bool viewport_restart_pending = false; // A full depth clear with no draw after it yet.
	bool viewport_restarted = false; // The extent covers only work after a full depth clear.
	// Execution-order positions of this frame's work (state_tracking::position,
	// rebased when a recording merges into the one it executes in); zero: none.
	// They are positions, not counters, so they are kept while counters are off.
	// last_overwrite: the last clear of the sampled plane's depth (with or
	// without rectangles, a render pass's CLEAR included) or D3D12 render pass
	// that DISCARDs it; what was there may be gone (resolve_transfer_lineage).
	uint64_t first_draw = 0, last_overwrite = 0;
	// The last transfer into this resource: its source (any resource) and
	// position, and whether it replaced the whole sampled plane.
	resource transfer_source = {};
	uint64_t transfer_position = 0;
	bool transfer_full = false;
	// This frame's transfer activity was removed (resolve_transfer_lineage):
	// it was a copy of a candidate that holds the same or newer depth.
	bool transfer_alias = false;

	void limit_tracking(depth_tracking_demand demand)
	{
		if (!demand.counters) total.clear_counters();
	}
	void observe_viewport(const viewport &view)
	{
		if (viewport_restart_pending && view.width * view.height > 0)
		{
			total.last_viewport = view;
			viewport_restart_pending = false;
			viewport_restarted = true;
		}
		else if (view.width * view.height >= total.last_viewport.width * total.last_viewport.height)
			total.last_viewport = view;
	}
	void observe_full_clear() { viewport_restart_pending = true; }
	// A depth clear of the sampled plane, with or without rectangles, overwrites
	// it (last_overwrite). Only without rectangles, and only on D3D10, D3D11
	// and D3D12 (ClearDepthStencilView, and a D3D12 render pass's CLEAR), does it
	// cover the whole view and so erase the extent drawn so far. D3D9's Clear
	// covers the current viewport and scissor, glClear the scissor and a Vulkan
	// render pass's CLEAR its render area, none of which ReShade reports: there
	// the crop keeps the largest viewport drawn in the frame.
	void observe_depth_clear(device_api api, uint32_t rect_count, bool sampled_plane, uint64_t position = 0)
	{
		if (!sampled_plane)
			return;
		last_overwrite = position;
		if (rect_count == 0 && (api == device_api::d3d10 || api == device_api::d3d11 || api == device_api::d3d12))
			observe_full_clear();
	}
	// A D3D12 render pass that DISCARDs the sampled plane's depth leaves it
	// undefined: it erases what was there as a clear does (the hold's cleared
	// gate, the lineage), but has no value, so it adds no clear convention and
	// does not restart the capture crop.
	void observe_depth_discard(uint64_t position)
	{
		depth_erased = true;
		last_overwrite = position;
	}
	// Raster or mesh work at this position.
	void observe_draw(uint64_t position)
	{
		if (!first_draw) first_draw = position;
	}
	void observe_transfer(resource source, uint64_t position, bool full)
	{
		transfer_source = source; transfer_position = position; transfer_full = full;
	}
	// Positions of a recording that executes after everything already merged.
	void merge_positions(const depth_stencil_frame_stats &later, uint64_t offset)
	{
		if (later.first_draw && !first_draw) first_draw = later.first_draw + offset;
		if (later.last_overwrite) last_overwrite = later.last_overwrite + offset;
		if (later.transfer_position)
			observe_transfer(later.transfer_source, later.transfer_position + offset, later.transfer_full);
		transfer_alias |= later.transfer_alias;
	}
};

struct resource_hash
{
	size_t operator()(resource value) const
	{
		// Simply use the handle (which is usually a pointer) as hash value (with some bits shaved off due to pointer alignment)
		return static_cast<size_t>(value.handle >> 4);
	}
};

struct __declspec(uuid("806c3b54-baf6-46a7-a72a-28c629143d84")) state_tracking
{
	const bool is_queue;
	const uint64_t command_identity = s_next_command_identity.fetch_add(1, std::memory_order_relaxed);
	viewport current_viewport = {};
	resource current_depth_stencil = { 0 };
	uint64_t current_depth_lifetime = 0;
	resource render_pass_depth = {};
	bool render_pass_depth_uncertain = false;
	// Execution order of the depth work recorded here (draws, clears and
	// transfers), for depth_stencil_frame_stats' positions.
	uint64_t ordinal = 0;
	std::unordered_map<resource, depth_stencil_frame_stats, resource_hash> stats_per_used_depth_stencil;
	explicit state_tracking(bool is_queue) : is_queue(is_queue)
	{
		// Reserve some space upfront to avoid rehashing during command recording
		stats_per_used_depth_stencil.reserve(32);
	}

	uint64_t position() { return ++ordinal; }
	void reset()
	{
		stats_per_used_depth_stencil.clear();
		current_depth_stencil = { 0 };
		current_depth_lifetime = 0;
		render_pass_depth = {};
		render_pass_depth_uncertain = false;
		ordinal = 0;
	}
	void reset_on_present()
	{
		assert(is_queue);
		stats_per_used_depth_stencil.clear();
		ordinal = 0;
	}

	void merge(const state_tracking &source, depth_tracking_demand demand = {})
	{
		// Executing a command list in a different command list inherits state
		current_depth_stencil = source.current_depth_stencil;
		current_depth_lifetime = source.current_depth_lifetime;

		if (source.stats_per_used_depth_stencil.empty())
			return;

		// The source's work executes after everything merged here so far.
		const uint64_t offset = ordinal;
		ordinal += source.ordinal;
		stats_per_used_depth_stencil.reserve(source.stats_per_used_depth_stencil.size());
		for (const auto &[depth_stencil, source_stats] : source.stats_per_used_depth_stencil)
		{
			depth_stencil_frame_stats &stats = stats_per_used_depth_stencil[depth_stencil];
			stats.merge_positions(source_stats, offset);
			stats.limit_tracking(demand);
			stats.total.observed_work |= source_stats.total.has_work();
			if (demand.counters)
			{
				stats.total.vertices += source_stats.total.vertices;
				stats.total.drawcalls += source_stats.total.drawcalls;
				stats.total.drawcalls_indirect += source_stats.total.drawcalls_indirect;
				stats.total.scene_writes += source_stats.total.scene_writes;
				stats.total.mesh_dispatches += source_stats.total.mesh_dispatches;
			}
			// The source executes after this recording: its own full clear erased
			// the extent drawn here, and a clear pending here restarts at its draws.
			if (source_stats.viewport_restarted)
			{
				stats.total.last_viewport = source_stats.total.last_viewport;
				stats.viewport_restart_pending = false;
				stats.viewport_restarted = true;
			}
			else
				stats.observe_viewport(source_stats.total.last_viewport);
			stats.viewport_restart_pending |= source_stats.viewport_restart_pending;
			stats.clear_zero |= source_stats.clear_zero;
			stats.clear_one |= source_stats.clear_one;
			stats.clear_other |= source_stats.clear_other;
			stats.depth_erased |= source_stats.depth_erased;
		}
	}
};

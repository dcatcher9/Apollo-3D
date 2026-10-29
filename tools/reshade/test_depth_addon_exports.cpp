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
#include "game3d_frame_record.h"
#include <chrono>
#include <stdexcept>
#include "depth_refusal.h"
#include "depth_selection_snapshot_test.h"

static void require_snapshot(bool value, const char *message)
{
	if (!value) throw std::runtime_error(message);
}

static auto snapshot_test_resources(unsigned count)
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
		frame.copied_viewport = { 16, 8, frame.total.last_viewport.width - 32, frame.total.last_viewport.height - 16, .1f, .9f };
		frame.copied_during_frame = i % 2;
		frame.copy_provenance_ambiguous = i % 3 == 0;
		if (i % 7 == 0) frame.total.last_viewport = {}; // Unknown draw/copy coverage.
		if (i % 5 == 0) frame.copied_viewport = {};
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
		x.copied_during_frame == y.copied_during_frame && x.copy_provenance_ambiguous == y.copy_provenance_ambiguous &&
		x.depth_erased == y.depth_erased && a.prior_work_frame == b.prior_work_frame,
		"Snapshot changed draw, mesh/copy work, viewport, copied-frame or hold-gate flags");
	require_snapshot(effective_capture_region(a) == effective_capture_region(b) &&
		candidate_matches_shape(a, 3840, 2160) == candidate_matches_shape(b, 3840, 2160),
		"Snapshot changed capture region or candidate shape admission");
}

// Candidate shape admission at the add-on's defaults. The expected values were
// recorded from the add-on built before its capture filter knobs were removed,
// not computed by hand; the fixed check must reproduce every row.
struct shape_row
{
	uint32_t width, height, frame_width, frame_height;
	bool expected;
};
static constexpr shape_row shape_rows[] = {
	// Zero sizes.
	{ 0, 0, 3840, 2160, true }, { 0, 2160, 3840, 2160, true }, { 3840, 0, 3840, 2160, true }, { 1920, 1080, 0, 0, false },
	// Native, DLSS Quality, Balanced, Performance and Ultra Performance, quarter, too small.
	{ 3840, 2160, 3840, 2160, true }, { 2560, 1440, 3840, 2160, true }, { 2227, 1253, 3840, 2160, true },
	{ 1920, 1080, 3840, 2160, true }, { 1280, 720, 3840, 2160, true }, { 960, 540, 3840, 2160, true },
	{ 640, 360, 3840, 2160, false },
	// Square and skinny shadow targets.
	{ 1024, 1024, 3840, 2160, false }, { 2048, 2048, 3840, 2160, false }, { 4096, 4096, 3840, 2160, false },
	{ 3840, 128, 3840, 2160, false },
	// Output/source ratios 1.85 and 0.5, exactly and just past.
	{ 1920, 1080, 3552, 1998, true }, { 1920, 1080, 3553, 1999, true }, { 2000, 1000, 3700, 1850, true },
	{ 1999, 1000, 3700, 1850, true }, { 3840, 2160, 1920, 1080, true }, { 3842, 2160, 1920, 1080, false },
	{ 3840, 2162, 1920, 1080, false }, { 1921, 1080, 3840, 2160, true },
	// Aspect delta about 0.099 and 0.101 at both ratio edges.
	{ 1096, 584, 1920, 1080, true }, { 1097, 584, 1920, 1080, false }, { 980, 584, 1920, 1080, true },
	{ 979, 584, 1920, 1080, false }, { 3626, 2160, 1920, 1080, true }, { 3622, 2160, 1920, 1080, false },
	{ 4054, 2160, 1920, 1080, false }, { 4058, 2160, 1920, 1080, false }, { 2192, 1168, 3840, 2160, true },
	{ 2194, 1168, 3840, 2160, false }, { 7252, 4320, 3840, 2160, true }, { 7244, 4320, 3840, 2160, false },
	// Aspect delta 0.1, where float and double rounding disagree: only the
	// float similar-aspect term admits the first two.
	{ 2472, 1080, 3440, 1440, true }, { 1260, 600, 2000, 1000, true }, { 906, 540, 1280, 720, true },
	// Ultrawide output.
	{ 2580, 1080, 3440, 1440, true }, { 2560, 1440, 3440, 1440, false },
};
enum class shape_region : uint8_t { copied, drawn, preserved };
struct shape_region_row
{
	uint32_t width, height;
	shape_region source;
	uint32_t x, y, region_width, region_height;
	bool expected;
};
// Padded allocations against a 3840x2160 frame; the last row's viewport
// exceeds the allocation, so the whole allocation is the region.
static constexpr shape_region_row shape_region_rows[] = {
	{ 4096, 4096, shape_region::copied, 0, 0, 2560, 1440, true },
	{ 4096, 4096, shape_region::drawn, 0, 0, 1920, 1080, true },
	{ 4096, 4096, shape_region::preserved, 0, 0, 1280, 720, true },
	{ 2048, 2048, shape_region::copied, 16, 8, 1920, 1080, true },
	{ 4096, 4096, shape_region::copied, 0, 0, 1024, 1024, false },
	{ 4096, 4096, shape_region::drawn, 0, 0, 5000, 3000, false },
};

static bool shape_admits(const depth_stencil_resource &info, uint32_t frame_width, uint32_t frame_height)
{
	const bool admitted = candidate_matches_shape(info, frame_width, frame_height);
	require_snapshot(admitted == candidate_matches_shape(depth_selection_resource(info), frame_width, frame_height),
		"Snapshot changed candidate shape admission");
	return admitted;
}

static void check_candidate_shape_table()
{
	char message[160];
	for (const auto &row : shape_rows)
	{
		depth_stencil_resource info;
		info.desc.type = resource_type::texture_2d;
		info.desc.texture = { row.width, row.height, 1, 1, format::d32_float, 1 };
		std::snprintf(message, sizeof(message), "Candidate shape admission changed for %ux%u at a %ux%u frame",
			row.width, row.height, row.frame_width, row.frame_height);
		require_snapshot(shape_admits(info, row.frame_width, row.frame_height) == row.expected, message);
	}
	for (const auto &row : shape_region_rows)
	{
		depth_stencil_resource info;
		info.desc.type = resource_type::texture_2d;
		info.desc.texture = { row.width, row.height, 1, 1, format::d32_float, 1 };
		const viewport view { float(row.x), float(row.y), float(row.region_width), float(row.region_height), 0, 1 };
		auto &stats = info.last_frame_stats;
		if (row.source == shape_region::copied)
		{
			stats.copied_during_frame = true;
			stats.copied_viewport = view;
		}
		else if (row.source == shape_region::drawn)
			stats.total.last_viewport = view;
		else
		{
			info.have_preserved_region = true;
			info.preserved_region = { row.x, row.y, row.region_width, row.region_height, info.layout_epoch };
		}
		std::snprintf(message, sizeof(message), "Candidate shape admission changed for a %ux%u region of a %ux%u allocation",
			row.region_width, row.region_height, row.width, row.height);
		require_snapshot(shape_admits(info, 3840, 2160) == row.expected, message);
	}
}

// The capture crop is the largest viewport drawn since the last full depth
// clear, in execution order across merged recordings (docs/reshade-sbs.md,
// Scene-depth selection limits).
static void check_capture_extent()
{
	const viewport full { 0, 0, 3840, 2160, 0, 1 }, three_quarters { 0, 0, 2880, 1620, 0, 1 },
		half { 0, 0, 1920, 1080, 0, 1 }, quarter { 0, 0, 960, 540, 0, 1 }, transfer {};
	const auto same = [](const viewport &a, const viewport &b) {
		return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
	};
	depth_stencil_frame_stats frame;
	frame.observe_full_clear(); frame.observe_viewport(full); frame.observe_viewport(half);
	require_snapshot(same(frame.total.last_viewport, full), "A smaller later pass narrowed the frame's capture crop");
	frame = {};
	frame.observe_viewport(three_quarters); frame.observe_full_clear(); frame.observe_viewport(transfer);
	require_snapshot(same(frame.total.last_viewport, three_quarters), "A clear or transfer without a later draw changed the capture crop");
	frame.observe_viewport(half); frame.observe_viewport(quarter);
	require_snapshot(same(frame.total.last_viewport, half) && frame.viewport_restarted && !frame.viewport_restart_pending,
		"A larger pass cleared before the scene still widened the scene's capture crop");
	depth_stencil_resource info;
	info.desc.type = resource_type::texture_2d;
	info.desc.texture = { 3840, 2160, 1, 1, format::d32_float, 1 };
	info.last_frame_stats = frame;
	const auto region = effective_capture_region(info);
	require_snapshot(region.x == 0 && region.y == 0 && region.width == 1920 && region.height == 1080,
		"The capture region did not follow the extent drawn after the last full clear");

	const resource target { 1 };
	const auto merged = [&](const std::initializer_list<void (*)(depth_stencil_frame_stats &)> &earlier,
		const std::initializer_list<void (*)(depth_stencil_frame_stats &)> &later) {
		state_tracking queue(true), list(false);
		for (const auto step : earlier) step(queue.stats_per_used_depth_stencil[target]);
		for (const auto step : later) step(list.stats_per_used_depth_stencil[target]);
		queue.merge(list);
		return queue.stats_per_used_depth_stencil[target];
	};
	static constexpr auto draw_full = +[](depth_stencil_frame_stats &s) { s.observe_viewport({ 0, 0, 3840, 2160, 0, 1 }); };
	static constexpr auto draw_half = +[](depth_stencil_frame_stats &s) { s.observe_viewport({ 0, 0, 1920, 1080, 0, 1 }); };
	static constexpr auto clear = +[](depth_stencil_frame_stats &s) { s.observe_full_clear(); };
	require_snapshot(same(merged({ draw_half }, { draw_full }).total.last_viewport, full),
		"Merged recordings without a clear lost the larger extent");
	require_snapshot(same(merged({ draw_full }, { clear, draw_half }).total.last_viewport, half),
		"A later recording's full clear did not erase the earlier extent");
	require_snapshot(same(merged({ draw_full, clear }, { draw_half }).total.last_viewport, half),
		"A pending clear did not restart at the next recording's draws");
	auto trailing = merged({ draw_full }, { draw_half, clear });
	require_snapshot(same(trailing.total.last_viewport, full) && trailing.viewport_restart_pending,
		"A later recording's trailing clear changed the extent or was lost");
	trailing.observe_viewport(quarter);
	require_snapshot(same(trailing.total.last_viewport, quarter), "A merged pending clear did not restart at the next draw");

	// A rect-less depth clear covers the whole view only on D3D10/11/12. D3D9's
	// Clear(0, NULL) clears the current viewport (a HUD model panel after the
	// scene), glClear the scissor and a Vulkan render pass's CLEAR its render
	// area: there the crop keeps the largest viewport drawn in the frame.
	for (const auto api : { device_api::d3d9, device_api::d3d10, device_api::d3d11, device_api::d3d12,
		device_api::opengl, device_api::vulkan })
	{
		const bool whole_view = api == device_api::d3d10 || api == device_api::d3d11 || api == device_api::d3d12;
		depth_stencil_frame_stats panel;
		panel.observe_depth_clear(api, 0, true); panel.observe_viewport(full);
		panel.observe_depth_clear(api, 0, true); panel.observe_viewport(quarter);
		require_snapshot(same(panel.total.last_viewport, whole_view ? quarter : full),
			whole_view ? "A whole-view depth clear did not restart the capture crop" :
				"A viewport-limited depth clear narrowed the scene's capture crop to a later small pass");
		depth_stencil_frame_stats rects;
		rects.observe_viewport(full); rects.observe_depth_clear(api, 1, true); rects.observe_viewport(quarter);
		depth_stencil_frame_stats other_plane;
		other_plane.observe_viewport(full); other_plane.observe_depth_clear(api, 0, false); other_plane.observe_viewport(quarter);
		require_snapshot(same(rects.total.last_viewport, full) && same(other_plane.total.last_viewport, full),
			"A clear limited to rectangles or another plane restarted the capture crop");
	}
}

// A copy of another same-geometry candidate is not depth of its own: a history
// copy made before the source drew holds older depth, and a copy the source
// still holds is a duplicate. Only a whole copy of a pass the source then
// cleared away keeps the destination (docs/reshade-sbs.md, Selecting scene depth).
static void check_transfer_lineage()
{
	const resource scene { 0x1000 }, history { 0x2000 }, msaa { 0x3000 }, small { 0x4000 }, buffer { 0x5000 };
	generic_depth_device_data device;
	const auto track = [&](resource value, uint32_t width, uint32_t height, uint16_t samples) {
		depth_stencil_resource info;
		info.desc.type = resource_type::texture_2d;
		info.desc.texture = { width, height, 1, 1, format::r32_typeless, samples };
		info.identity = value.handle;
		device.depth_stencil_resources.emplace(value, info);
	};
	track(scene, 1920, 1080, 1); track(history, 1920, 1080, 1); track(msaa, 1920, 1080, 4); track(small, 960, 540, 1);
	enum class op { draw, clear, copy, partial };
	struct step
	{
		op kind;
		resource target, source;
		step(op k, resource t, resource s = {}) : kind(k), target(t), source(s) {}
	};
	// Records the steps into consecutive recordings (one per group) executed in
	// order, as on_execute_primary merges them, then resolves the queue.
	const auto run = [&](const std::initializer_list<std::initializer_list<step>> &groups) {
		state_tracking queue(true);
		for (const auto &group : groups)
		{
			state_tracking list(false);
			for (const auto &s : group)
			{
				auto &stats = list.stats_per_used_depth_stencil[s.target];
				if (s.kind == op::draw) { stats.total.observed_work = true; ++stats.total.drawcalls; stats.observe_draw(list.position()); }
				else if (s.kind == op::clear) { stats.clear_zero = true; stats.observe_depth_clear(device_api::d3d12, 0, true, list.position()); }
				else
				{
					stats.total.observed_work = true; ++stats.total.scene_writes;
					stats.observe_transfer(s.source, list.position(), s.kind == op::copy);
				}
			}
			queue.merge(list);
		}
		resolve_transfer_lineage(queue, device);
		return queue.stats_per_used_depth_stencil;
	};
	const auto active = [](const auto &stats, resource value) { return stats.count(value) && stats.at(value).total.has_work(); };
	auto frame = run({{{op::copy, history, scene}, {op::clear, scene}, {op::draw, scene}}});
	require_snapshot(!frame.count(history) && active(frame, scene),
		"A history copy made before its source drew stayed current depth, or left its destination touched");
	frame = run({{{op::clear, history}, {op::copy, history, scene}, {op::clear, scene}, {op::draw, scene}}});
	require_snapshot(frame.count(history) && !active(frame, history) && frame.at(history).transfer_alias && frame.at(history).clear_zero,
		"A history reset (clear, then copy) lost its clear or kept its copy as activity");
	frame = run({{{op::clear, scene}, {op::draw, scene}}, {{op::copy, history, scene}}});
	require_snapshot(!active(frame, history) && active(frame, scene), "A duplicate of depth its source still holds stayed current depth");
	frame = run({{{op::clear, scene}, {op::draw, scene}, {op::copy, history, scene}}, {{op::clear, scene}, {op::draw, scene}}});
	const auto &kept = frame.at(history);
	require_snapshot(active(frame, history) && !kept.transfer_alias && kept.clear_zero,
		"A whole copy of a pass its source cleared away lost its activity or the source's clear convention");
	frame = run({{{op::clear, scene}, {op::draw, scene}, {op::partial, history, scene}, {op::clear, scene}, {op::draw, scene}}});
	require_snapshot(!active(frame, history), "A partial copy of a pass was preferred over a source that drew");
	frame = run({{{op::copy, history, scene}}});
	require_snapshot(!active(frame, history), "A copy of a source with no work in this queue stayed current depth");
	frame = run({{{op::draw, msaa}, {op::copy, history, msaa}}, {{op::draw, small}, {op::copy, history, small}}});
	require_snapshot(active(frame, history), "A copy from a non-candidate or another geometry lost its activity");
	frame = run({{{op::draw, msaa}, {op::copy, history, scene}, {op::copy, history, buffer}}});
	require_snapshot(active(frame, history), "A later copy from a non-depth source lost its activity");
	frame = run({{{op::copy, history, scene}, {op::draw, history}, {op::draw, scene}}});
	require_snapshot(active(frame, history), "A destination that also drew lost its activity");
}

// The hold gates (docs/reshade-sbs.md, Selecting scene depth): a Present that
// cleared or discarded the anchor's capture set without any of it working, or
// on which a shape-eligible depth-stencil outside the set drew without having
// worked within the selector's activity grace, holds nothing. Everything else
// that the held set left untouched, or worked on, may hold.
static void check_hold_gates()
{
	enum class work { none, draw, clear, clear_draw, discard, copy };
	struct source
	{
		uint32_t width, height;
		uint16_t samples;
		std::vector<work> frames; // Frames 1.. in order; none = not used.
	};
	constexpr uint32_t color_width = 1920, color_height = 1080;
	const auto stats_of = [](work kind) {
		depth_stencil_frame_stats stats;
		if (kind == work::draw || kind == work::clear_draw) { stats.total.observed_work = true; ++stats.total.drawcalls; }
		if (kind == work::copy) { stats.total.observed_work = true; ++stats.total.scene_writes; }
		if (kind == work::clear || kind == work::clear_draw) { stats.clear_zero = true; stats.depth_erased = true; }
		if (kind == work::discard) stats.depth_erased = true;
		return stats;
	};
	// Plays every source's frames through on_present's bookkeeping, then
	// judges the last frame with `set` as the capture set and `held` held.
	const auto judge = [&](const std::vector<source> &sources, const std::vector<unsigned> &set, unsigned held = 1, bool resolve = false) {
		std::unordered_map<resource, depth_stencil_resource, resource_hash> resources;
		uint64_t frames = 0;
		for (unsigned i = 0; i != sources.size(); ++i)
		{
			depth_stencil_resource info;
			info.desc.type = resource_type::texture_2d;
			info.desc.texture = { sources[i].width, sources[i].height, 1, 1, format::d32_float, sources[i].samples };
			info.identity = i + 1;
			for (unsigned frame = 0; frame != sources[i].frames.size(); ++frame)
				if (sources[i].frames[frame] != work::none) info.observe_frame(stats_of(sources[i].frames[frame]), frame + 1);
			frames = std::max<uint64_t>(frames, sources[i].frames.size());
			resources.emplace(resource { 0x1000 + uint64_t(i) * 16 }, info);
		}
		sunshine_depth_capture::capture_set captures;
		for (const unsigned id : set) captures.records[captures.count++].identity = id;
		const auto snapshot = snapshot_depth_selection(resources);
		return judge_hold_gate(snapshot, captures, held, frames, color_width, color_height, resolve,
			sunshine_depth::selection_policy {}.active_grace_frames());
	};
	using w = work;
	const source anchor { 1920, 1080, 1, { w::draw, w::draw, w::draw } };
	// Untouched, or worked on: the hold stays open.
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 1, { w::draw, w::draw, w::draw } } }, { 1 }) == hold_gate::open,
		"A HUD drawing every Present into its own depth-stencil refused the anchor's hold");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 960, 540, 1, { w::draw, w::draw, w::draw } } }, { 1 }) == hold_gate::open,
		"An every-Present lower-resolution source refused the hold of the frames the anchor skips");
	require_snapshot(judge({ anchor }, { 1 }) == hold_gate::open, "A set that drew (a refused or failed copy, a missing frame) refused its hold");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::clear_draw } } }, { 1 }) == hold_gate::open,
		"A set that cleared and drew refused its hold");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::copy } } }, { 1 }) == hold_gate::open,
		"A set that received its own copy refused its hold");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::none, w::none } }, { 1920, 1080, 1, { w::none, w::draw, w::none } },
		{ 1920, 1080, 1, { w::draw, w::draw, w::draw } } }, { 1, 2 }, 2) == hold_gate::open,
		"An alternating set left untouched on a Present with other depth work refused its hold");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::clear } }, { 1920, 1080, 1, { w::none, w::none, w::draw } } }, { 1, 2 }) == hold_gate::open,
		"A set member that cleared while another drew refused the hold");
	// Cleared without any of the set working: a screen without its scene.
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::clear } } }, { 1 }) == hold_gate::cleared,
		"A Present that cleared the anchor and drew nothing held its last scene");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::discard } } }, { 1 }) == hold_gate::cleared,
		"A Present that discarded the anchor's depth and drew nothing held its last scene");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 1, { w::none, w::draw, w::clear } } }, { 1, 2 }) == hold_gate::cleared,
		"A cleared alternating set member with nothing drawn left the hold open");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::clear } } }, {}) == hold_gate::cleared,
		"Clearing the held source outside the routed set left the hold open");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::clear } }, { 1920, 1080, 1, { w::draw, w::draw, w::draw } } }, { 1 }) == hold_gate::cleared,
		"An every-Present HUD drawing beside a cleared anchor left the hold open");
	// A scene pass that starts now in another buffer: another camera's depth.
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 1, { w::none, w::none, w::draw } } }, { 1 }) == hold_gate::cut,
		"A same-size unqualified scene starting in another allocation held the previous scene");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1280, 720, 1, { w::none, w::none, w::draw } } }, { 1 }) == hold_gate::cut,
		"A scene starting in another-size allocation held the previous scene");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::draw, w::draw, w::draw, w::none } },
		{ 1920, 1080, 1, { w::draw, w::none, w::none, w::none, w::none, w::draw } } }, { 1 }) == hold_gate::cut,
		"A source drawing again after more than the activity grace without work held the previous scene");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 1, { w::clear, w::clear, w::draw } } }, { 1 }) == hold_gate::cut,
		"A source that only cleared on the previous frame and now draws held the previous scene");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 4, { w::none, w::none, w::draw } } }, { 1 }, 1, true) == hold_gate::cut,
		"A resolvable multisampled scene starting now held the previous scene");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 4, { w::none, w::none, w::draw } } }, { 1 }) == hold_gate::open,
		"A multisampled buffer that cannot be resolved (no candidate) refused the hold");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1024, 1024, 1, { w::none, w::none, w::draw } } }, { 1 }) == hold_gate::open,
		"A buffer of another aspect (a shadow map) refused the hold");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 1, { w::none, w::none, w::copy } } }, { 1 }) == hold_gate::open,
		"A copy into another buffer was taken for a scene pass");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 1, { w::none, w::none, w::draw } } }, { 1, 2 }) == hold_gate::open,
		"A member of the anchor's own set was taken for a cut");
	require_snapshot(judge({ { 1920, 1080, 1, { w::none, w::draw, w::none, w::none } }, { 1920, 1080, 1, { w::none, w::none, w::draw, w::none } },
		{ 1920, 1080, 1, { w::draw, w::none, w::none, w::draw } }, { 960, 540, 1, { w::draw, w::draw, w::draw, w::draw } } }, { 1, 2 }, 2) == hold_gate::open,
		"An unqualified member of an A/B/C rotation on its turn was taken for a cut");
	require_snapshot(judge({ { 1920, 1080, 1, { w::draw, w::draw, w::none } }, { 1920, 1080, 1, { w::none, w::draw, w::draw } } }, { 1 }) == hold_gate::open,
		"A source that also drew on the previous depth frame was taken for a cut");
	// Rotations as long as a capture set holds (A/B, A/B/C, A/B/C/D: periods 2
	// to 4, which the selector's activity grace keeps active between turns).
	// The member on its turn is no cut, inside the anchor's set or not yet
	// qualified into it, with the previous member's capture held; a scene
	// starting in another buffer beside the untouched set is one; the member
	// on its turn clearing without drawing is a screen without its scene.
	for (unsigned period = 2; period <= 4; ++period)
	{
		std::vector<source> rotation;
		std::vector<unsigned> members, qualified;
		for (unsigned member = 0; member != period; ++member)
		{
			source turns { 1920, 1080, 1, {} };
			for (unsigned frame = 0; frame != 3 * period; ++frame)
				turns.frames.push_back(frame % period == member ? w::draw : w::none);
			rotation.push_back(turns);
			members.push_back(member + 1);
			if (member + 1 != period) qualified.push_back(member + 1);
		}
		// The last frame is the last member's turn; the member before it showed the held capture.
		require_snapshot(judge(rotation, members, period - 1) == hold_gate::open,
			"A member of the anchor's A/B, A/B/C or A/B/C/D rotation on its turn refused the hold");
		require_snapshot(judge(rotation, qualified, period - 1) == hold_gate::open,
			"An unqualified member of an A/B, A/B/C or A/B/C/D rotation on its turn was taken for a cut");
		auto cut = rotation;
		for (auto &turns : cut) turns.frames.push_back(w::none);
		cut.push_back({ 1920, 1080, 1, std::vector<work>(3 * period, w::none) });
		cut.back().frames.push_back(w::draw);
		require_snapshot(judge(cut, members, period) == hold_gate::cut,
			"A scene starting in another buffer beside an untouched rotation held the rotation's depth");
		auto cleared = rotation;
		cleared.back().frames.back() = w::clear;
		require_snapshot(judge(cleared, members, period - 1) == hold_gate::cleared,
			"A rotation member clearing on its turn without drawing held the previous member's depth");
	}
	// on_present's bookkeeping: the last earlier frame with work.
	constexpr uint64_t none = std::numeric_limits<uint64_t>::max();
	depth_stencil_resource info;
	info.observe_frame(stats_of(w::draw), 4);
	require_snapshot(info.prior_work_frame == none && info.first_used_in_frame == 4 && info.last_used_in_frame == 4,
		"A first use claimed earlier work");
	info.observe_frame(stats_of(w::clear), 5);
	require_snapshot(info.prior_work_frame == 4, "Work on the previous depth frame was not kept");
	info.observe_frame(stats_of(w::draw), 6);
	require_snapshot(info.prior_work_frame == 4, "A frame with only a clear counted as work");
	info.observe_frame(stats_of(w::draw), 9);
	require_snapshot(info.prior_work_frame == 6 && info.first_used_in_frame == 4, "The last earlier frame with work was not kept");
	depth_stencil_resource cleared;
	cleared.observe_frame(stats_of(w::clear), 2);
	cleared.observe_frame(stats_of(w::draw), 3);
	require_snapshot(cleared.prior_work_frame == none, "A clear-only first use counted as work");
	// The erased mark survives merging recordings into the queue.
	state_tracking queue(true), list(false);
	list.stats_per_used_depth_stencil[resource { 0x1000 }].depth_erased = true;
	queue.merge(list);
	require_snapshot(queue.stats_per_used_depth_stencil[resource { 0x1000 }].depth_erased, "Merging lost a depth clear or discard");
}

static void check_depth_snapshot_contract()
{
	check_candidate_shape_table();
	check_capture_extent();
	check_transfer_lineage();
	check_hold_gates();

	// Candidate counters follow the capture-owner count (or the opt-in trace).
	generic_depth_device_data demand_device;
	require_snapshot(tracking_demand(demand_device).counters == s_frame_activity_trace,
		"Hidden API-native path retained candidate demand");
	demand_device.generic_capture_users.store(1);
	require_snapshot(tracking_demand(demand_device).counters,
		"Generic/shared preservation lost numeric capture statistics");

	// An already recorded command list may execute after candidate demand
	// stops. Merge preserves transport evidence, never its old counters.
	const resource tracked {0x1000};
	state_tracking cached(false), merged(false);
	auto &cached_stats = cached.stats_per_used_depth_stencil[tracked];
	cached_stats.total = {1200, 12, 3, 2, {0, 0, 1920, 1080, 0, 1}};
	cached_stats.copied_during_frame = cached_stats.clear_zero = true;
	cached_stats.copied_viewport = cached_stats.total.last_viewport;
	cached_stats.capture_ticket.id = 77;
	cached_stats.capture_ticket.texture = 88;
	merged.merge(cached); // Existing detailed accumulation must be cleared too.
	merged.merge(cached, {false});
	const auto &paused = merged.stats_per_used_depth_stencil.at(tracked);
	require_snapshot(paused.total.has_work() && paused.total.work_count() == 0 &&
		paused.total.vertices == 0 && paused.total.drawcalls_indirect == 0,
		"Cached command lists repopulated paused candidate counters");
	require_snapshot(paused.copied_during_frame && paused.clear_zero &&
		paused.capture_ticket.id == 77 && paused.capture_ticket.texture == 88 &&
		same_snapshot_viewport(paused.total.last_viewport, cached_stats.total.last_viewport) &&
		same_snapshot_viewport(paused.copied_viewport, cached_stats.copied_viewport),
		"Pausing candidate details changed preservation eligibility, layout or capture ownership");
	state_tracking resumed(false);
	resumed.merge(merged, {true});
	require_snapshot(resumed.stats_per_used_depth_stencil.at(tracked).total.has_work() &&
		resumed.stats_per_used_depth_stencil.at(tracked).total.work_count() == 0,
		"Resuming counters invented workload for a recording observed while details were paused");
	resumed.merge(cached, {true});
	require_snapshot(resumed.stats_per_used_depth_stencil.at(tracked).total.vertices == 1200 &&
		resumed.stats_per_used_depth_stencil.at(tracked).total.drawcalls == 12,
		"Resumed Generic/shared preservation changed known workload");

	auto resources = snapshot_test_resources(16);
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
	resources.begin()->second.last_used_in_frame = 90; // A live inactive candidate must survive the snapshot.
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
		"Snapshot filtered out a live inactive candidate");
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
	// pass. A live inactive candidate is retained; a destroyed and a filtered
	// row are erased before delayed readbacks/selection can see their old evidence.
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
		"Live inactive candidate lost its retained histogram or calibration");
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

extern "C" __declspec(dllexport) BOOL SunshineDepthTestSnapshot(unsigned buffers, unsigned iterations,
	sunshine_depth_snapshot_test_result *output)
{
	if (!output) return FALSE;
	*output = {};
	try
	{
		require_snapshot(buffers <= 512 && iterations && iterations <= 100000, "Invalid benchmark workload");
		check_depth_snapshot_contract();
		const auto resources = snapshot_test_resources(buffers);
		output->deep_value_bytes = buffers * sizeof(decltype(resources)::value_type);
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

// The test adapters read add-on state the overlay shows; they never change
// selection or inject depth pixels, and the production DLL omits them.
extern "C" __declspec(dllexport) BOOL SunshineDepthTestProviderStatus(effect_runtime *runtime,
	sunshine_streamline::provider::source_status *output)
{
	if (!s_registered || runtime == nullptr || output == nullptr) return FALSE;
	*output = describe_streamline_source(runtime);
	return TRUE;
}

// A copy of the frame record as the present thread left it. Nothing in the
// add-on reads the record; the provider runtime test compares it with the log
// and judges the continuity invariant with it (test_game3d_continuity.h).
extern "C" __declspec(dllexport) BOOL SunshineGame3DTestFrameRecord(effect_runtime *runtime,
	sunshine_game3d::frame_record *output)
{
	if (!s_registered || runtime == nullptr || output == nullptr) return FALSE;
	const auto *record = sunshine_game3d::frame_record_store::writable(runtime);
	if (record == nullptr) return FALSE;
	*output = *record;
	return TRUE;
}

// The newest depth refusal of the Streamline, NGX or DLSS-G adapters or the
// capture owner, with its contract class (depth_refusal.h). It is recorded
// where the refusal is decided, before any log rate limit, so a runtime case
// reads its reason at once instead of waiting for a log line. The adapters
// are process-wide; serial tells a new refusal from an earlier one.
extern "C" __declspec(dllexport) BOOL SunshineDepthTestLastRefusal(effect_runtime *runtime, sunshine_depth_refusal::record *output)
{
	if (!s_registered || runtime == nullptr || output == nullptr) return FALSE;
	*output = sunshine_depth_refusal::latest();
	return TRUE;
}

// Whether a native resource is in the Generic depth inventory and was used in
// the current frame. Read-only: it never changes selection.
extern "C" __declspec(dllexport) BOOL SunshineDepthTestTracked(effect_runtime *runtime, uint64_t native_resource, BOOL *tracked)
{
	if (!s_registered || runtime == nullptr || tracked == nullptr)
		return FALSE;
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (runtime->get_private_data<generic_depth_data>() == nullptr || device_data == nullptr)
		return FALSE;
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	const auto found = device_data->depth_stencil_resources.find(resource { native_resource });
	*tracked = found != device_data->depth_stencil_resources.end() &&
		found->second.last_used_in_frame == device_data->frame_index ? TRUE : FALSE;
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
		~restore_state() {
			const std::unique_lock<std::shared_mutex> lock(s_mutex);
			state.stats_per_used_depth_stencil = std::move(stats);
			state.current_depth_stencil = depth; state.current_depth_lifetime = lifetime;
			state.current_viewport = view;
		}
	};
	std::unique_lock<std::shared_mutex> lock(s_mutex);
	restore_state saved {state, std::move(state.stats_per_used_depth_stencil), state.current_depth_stencil,
		state.current_depth_lifetime, state.current_viewport};
	state.stats_per_used_depth_stencil.clear();
	state.current_depth_stencil = {native_resource}; state.current_depth_lifetime = 0;
	state.current_viewport = {0, 0, 1280, 720, 0, 1};
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
	ok &= state.stats_per_used_depth_stencil.empty();
	if (!checkpoint("Depth activity test: no-op accounting failed")) return FALSE;
	ok &= !on_dispatch_mesh(cmd, 2, 3, 4);
	auto &mesh = state.stats_per_used_depth_stencil[target];
	ok &= mesh.total.has_work() && mesh.total.scene_writes == 1 && mesh.total.mesh_dispatches == 1 && mesh.total.drew() &&
		mesh.total.drawcalls == 0 && mesh.total.vertices == 0 && mesh.total.last_viewport.width == 1280;
	state_tracking merged(false); merged.merge(state);
	ok &= merged.stats_per_used_depth_stencil[target].total.scene_writes == 1 &&
		merged.stats_per_used_depth_stencil[target].total.mesh_dispatches == 1;
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
	ok &= copied.total.scene_writes == 4 && copied.total.has_work() &&
		copied.total.drawcalls == 0 && copied.total.vertices == 0 && copied.total.last_viewport.width == 0 &&
		copied.total.mesh_dispatches == 0 && !copied.total.drew(); // A transfer is activity, not a drawn Present.
	if (!checkpoint("Depth activity test: base-depth copy accounting failed")) return FALSE;
	if (!s_frame_activity_trace)
	{
		auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
		if (device_data == nullptr) return FALSE;
		struct restore_demand {
			generic_depth_device_data &data;
			unsigned captures;
			~restore_demand() { data.generic_capture_users.store(captures); }
		} demand {*device_data, device_data->generic_capture_users.exchange(0)};
		state.stats_per_used_depth_stencil.clear();
		ok &= !on_draw(cmd, 3, 1, 0, 0) && !on_draw_indirect(cmd, indirect_command::draw, {}, 0, 2, 0) &&
			!on_dispatch_mesh(cmd, 1, 1, 1) && !on_copy_resource(cmd, {}, target);
		auto &passive = state.stats_per_used_depth_stencil[target];
		ok &= passive.total.has_work() && passive.total.work_count() == 0 &&
			passive.total.vertices == 0 && passive.total.drawcalls_indirect == 0 && passive.total.last_viewport.width == 1280;
		device_data->generic_capture_users.store(1);
		ok &= !on_draw(cmd, 3, 1, 0, 0) && passive.total.drawcalls == 1 && passive.total.vertices == 3;
		if (!checkpoint("Depth activity test: hidden API counters/Generic capture resume failed")) return FALSE;
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

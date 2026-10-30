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

// Generic depth's capture routing, a part of depth_addon.cpp (included where
// these definitions stood, after the candidate-selection declarations, in the
// same translation unit): the FrameActivityTrace text and Present/access
// trace types, member views, capture records and their validation, the hold
// gates (judge_hold_gate) and hold (hold_shown_capture), Present coverage, the
// end-of-frame copies (preserve_depth_at_effects) and select_captured_depth,
// whose FrameActivityTrace route block is generic_trace.inl.
#pragma once

// Temporary, bounded metadata diagnostics. No resource is read, retained or
// transitioned here. Logging happens after the caller's resource-map lock exits.
struct activity_trace_text
{
	char text[4096] {};
	size_t size = 0;
	template <typename... Args> void append(const char *format, Args... args)
	{
		if (size >= std::size(text) - 1) return;
		const int count = std::snprintf(text + size, std::size(text) - size, format, args...);
		if (count > 0) size += std::min(static_cast<size_t>(count), std::size(text) - size - 1);
	}
};

struct activity_present_trace
{
	struct row
	{
		uint64_t queue = 0, queue_native = 0, source = 0, handle = 0, backup = 0;
		uint32_t width = 0, height = 0, draws = 0, writes = 0;
		bool copied = false, ambiguous = false;
	};
	swapchain *chain;
	command_queue *presenting_queue;
	std::array<row, 16> rows {};
	size_t used = 0, total_rows = 0, queues = 0;
	uint64_t device_epoch = 0, present = 0, frame_before = 0, frame_after = 0, depth_present = 0, focus = 0;
	unsigned burst = 0, previous_dropped_events = 0;
	bool enabled = false;
	activity_present_trace(swapchain *value, command_queue *queue) : chain(value), presenting_queue(queue) {}

	void begin(const generic_depth_device_data &device_data)
	{
		const auto &trace = device_data.activity_trace;
		enabled = s_frame_activity_trace && trace.bursts &&
			device_data.native_present_index > trace.first_present && device_data.native_present_index <= trace.last_present;
		if (!enabled) return;
		device_epoch = device_data.device_epoch; burst = trace.bursts;
		present = device_data.native_present_index; frame_before = device_data.frame_index;
		frame_after = frame_before; depth_present = device_data.native_depth_present_index;
		focus = trace.focus_source; previous_dropped_events = trace.dropped_events;
	}
	void add(command_queue *queue, const state_tracking &state, generic_depth_device_data &device_data)
	{
		if (!enabled || state.stats_per_used_depth_stencil.empty()) return;
		++queues;
		for (const auto &[resource, stats] : state.stats_per_used_depth_stencil)
		{
			++total_rows;
			const auto found = device_data.depth_stencil_resources.find(resource);
			const uint64_t id = found == device_data.depth_stencil_resources.end() ? 0 : found->second.identity;
			size_t slot = used;
			if (slot == rows.size())
			{
				if (id != focus) continue;
				for (slot = 0; slot != rows.size() && rows[slot].source == focus; ++slot) {}
				if (slot == rows.size()) continue;
			}
			else ++used;
			auto &out = rows[slot];
			out = {};
			out.queue = queue->get_private_data<state_tracking>()->command_identity;
			out.queue_native = queue->get_native();
			out.source = id; out.handle = resource.handle;
			if (found != device_data.depth_stencil_resources.end())
			{
				out.width = found->second.desc.texture.width; out.height = found->second.desc.texture.height;
			}
			if (const auto *backup = device_data.find_depth_stencil_backup(resource)) out.backup = backup->content_identity;
			out.draws = stats.total.drawcalls; out.writes = stats.total.scene_writes;
			out.copied = stats.copied_during_frame; out.ambiguous = stats.copy_provenance_ambiguous;
		}
	}
	~activity_present_trace()
	{
		if (!enabled) return;
		activity_trace_text out;
		out.append("Sunshine frame activity: present burst=%u device=%llu p=%llu f=%llu->%llu depth_p=%llu tick=%llu thread=%lu swap=%llx queue=%llx bb_index=%u focus=%llu queues=%zu rows=%zu dropped_rows=%zu previous_dropped_events=%u;",
			burst, device_epoch, present, frame_before, frame_after, depth_present, GetTickCount64(), GetCurrentThreadId(),
			chain->get_native(), presenting_queue ? presenting_queue->get_native() : 0,
			chain->get_current_back_buffer_index(), focus,
			queues, used, total_rows - used, previous_dropped_events);
		for (size_t i = 0; i != used; ++i)
		{
			const auto &r = rows[i];
			out.append(" q%llu@%llx:s%llu@%llx,%ux%u,d%u,w%u,c%d,a%d,b%llu;", r.queue, r.queue_native, r.source, r.handle,
				r.width, r.height, r.draws, r.writes, r.copied ? 1 : 0, r.ambiguous ? 1 : 0, r.backup);
		}
		sunshine_log::message(reshade::log::level::info, out.text);
	}
};

static void trace_activity_effect(effect_runtime *runtime, generic_depth_data &data,
	generic_depth_device_data &device_data, bool access, const char *reason,
	uint64_t decision_present = 0, uint64_t decision_frame = 0)
{
	if (!s_frame_activity_trace) return;
	const auto *evidence = data.selection.evidence(data.anchor_identity);
	const bool confirmed = evidence && evidence->confirmed_good && evidence->quality.kind == sunshine_depth::content_kind::useful;
	activity_trace_text out;
	{
		const std::unique_lock<std::shared_mutex> lock(s_mutex);
		auto &trace = device_data.activity_trace;
		const auto now = GetTickCount64(), present = device_data.native_present_index;
		// A physical member change (or selected=0 on rejection) is not a new
		// capture set. Observe gaps against the stable budget generation and anchor.
		if (!access)
			data.trace_gaps.observe(present, data.raw_roster.routing_epoch,
				data.raw_scene_requested && confirmed && data.anchor_identity != 0, data.capture_ready);
		const unsigned misses = data.trace_gaps.misses();
		bool started = false;
		if (!access && data.raw_scene_requested && confirmed && data.anchor_identity && !data.capture_ready &&
			data.trace_gaps.recurring() &&
			present > trace.last_present && trace.bursts < 4 && now >= trace.next_start_tick)
		{
			++trace.bursts;
			trace.first_present = present; trace.last_present = present + 96;
			trace.next_start_tick = now + 15000;
			started = true;
		}
		if (!trace.bursts || present < trace.first_present || present > trace.last_present) return;
		if (trace.event_present != present)
		{
			trace.event_present = present;
			trace.effects = trace.accesses = trace.dropped_events = 0;
		}
		auto &last = access ? data.trace_access_present : data.trace_effect_present;
		if (last == present) return; // Only the first accessor/effects call per runtime/present.
		last = present;
		auto &count = access ? trace.accesses : trace.effects;
		if (count++ >= 2) { ++trace.dropped_events; return; }
		trace.focus_source = data.anchor_identity;
		uint64_t last_frame = 0, work = 0, backup = 0, layout = 0;
		bool copied = false, ambiguous = false;
		if (const auto found = device_data.depth_stencil_resources.find(data.selected_depth_stencil);
			found != device_data.depth_stencil_resources.end() && found->second.identity == data.selected_identity)
		{
			last_frame = found->second.last_used_in_frame; work = found->second.last_frame_stats.total.work_count();
			layout = found->second.layout_epoch;
			copied = found->second.last_frame_stats.copied_during_frame; ambiguous = found->second.last_frame_stats.copy_provenance_ambiguous;
		}
		if (const auto *tracked = device_data.find_depth_stencil_backup(data.selected_depth_stencil)) backup = tracked->content_identity;
		out.append("Sunshine frame activity: %s burst=%u start=%d device=%llu runtime=%llu snapshot=%llu/%llu decision=%llu/%llu final=%llu/%llu depth_p=%llu access_p=%llu tick=%llu thread=%lu source=%llu layout=%llu last_f=%llu work=%llu copy=%d ambiguous=%d backup=%llu pending=%llu probe=%llu ready=%d open=%d confirmed=%d misses16=%u reason=%s",
			access ? "access" : "effects", trace.bursts, started ? 1 : 0, device_data.device_epoch, data.runtime_epoch,
			data.trace_snapshot_present, data.trace_snapshot_frame, decision_present, decision_frame, present, device_data.frame_index,
			device_data.native_depth_present_index, data.native_access_present, now, GetCurrentThreadId(), data.selected_identity,
			layout, last_frame, work, copied ? 1 : 0, ambiguous ? 1 : 0, backup, data.selection.pending_challenger_id(), data.probe_target_id,
			data.capture_ready ? 1 : 0, data.native_access_open ? 1 : 0, confirmed ? 1 : 0, misses, reason);
		out.append(" challenger=%llu probe_age_ms=%llu challenger_samples=%u retirement_remaining_frames=%llu",
			data.challenger_id, data.probe_target_id && now >= data.probe_started_tick ? now - data.probe_started_tick : 0,
			data.challenger_samples, data.challenger_retired_until > device_data.frame_index ?
				data.challenger_retired_until - device_data.frame_index : 0);
		const auto &routing = data.trace_capture;
		out.append(" route=%llu/%llu anchor=%llu budget=%llu members=%u raster_mask=%x copy_mask=%x ambiguous_mask=%x confirmed_mask=%x view_mask=%x floor_mask=%x route_reason=%s;",
			routing.present, routing.frame, routing.anchor, routing.epoch, routing.count, routing.raster, routing.copied,
			routing.ambiguous, routing.confirmed, routing.views, routing.floors, routing.reason);
		for (unsigned i = 0; i != routing.count; ++i)
		{
			const auto &region = routing.regions[i];
			out.append(" m%u=s%llu/l%llu/f%llu/d%llu/w%llu/crop%u,%u,%u,%u;", i, routing.members[i], region.layout_epoch,
				routing.last_used[i], routing.draws[i], routing.writes[i], region.x, region.y, region.width, region.height);
		}
	}
	// Use scalar SDK getters only: aggregate-by-value resource getters cross
	// the MSVC/MinGW ABI. Back-buffer index is sufficient for this passive trace.
	out.append(" swap=%llx queue=%llx bb_index=%u", runtime->get_native(), runtime->get_command_queue()->get_native(),
		runtime->get_current_back_buffer_index());
	sunshine_log::message(reshade::log::level::info, out.text);
}

struct activity_access_trace
{
	effect_runtime *runtime;
	generic_depth_data *data;
	generic_depth_device_data *device_data;
	const char *reason = "ready";
	uint64_t decision_present = 0, decision_frame = 0;
	bool reject(const char *value) { reason = value; return false; }
	~activity_access_trace()
	{
		if (s_frame_activity_trace && data && device_data)
			trace_activity_effect(runtime, *data, *device_data, true, reason, decision_present, decision_frame);
	}
};

static bool owns_member_view(const generic_depth_data &data, resource_view view)
{
	return view != 0 && std::any_of(data.member_views.begin(), data.member_views.end(),
		[view](const auto &member) { return member.view == view; });
}

static void retire_member_views(effect_runtime *runtime, generic_depth_data &data,
	generic_depth_device_data &device_data, bool destroy_all = false)
{
	// No map lock: releasing either object can call resource-destruction hooks.
	for (auto &member : data.member_views)
		if (member.view != 0 && (destroy_all || (member.retire_after && device_data.frame_index >= member.retire_after)))
		{
			if (!destroy_all && member.view == data.selected_shader_resource) continue; // Unbind before destroying the current SRV.
			runtime->get_device()->destroy_resource_view(member.view);
			device_data.untrack_depth_stencil(runtime->get_device(), member.source);
			member = {};
		}
}

static generic_depth_data::raw_member_sample *raw_member(generic_depth_data &data, uint64_t id, bool create = false)
{
	if (!id) return nullptr;
	for (auto &entry : data.raw_members) if (entry.identity == id) return &entry;
	if (create)
		for (auto &entry : data.raw_members) if (!entry.identity) { entry.identity = id; return &entry; }
	return nullptr;
}

struct captured_depth
{
	sunshine_depth_capture::capture_record record;
	sunshine_depth::frame_depth metadata;
	bool end_of_frame = false; // The frame's copy is the end-of-frame one (counted only).
};

// One authority for callback-scoped current-copy facts. Source/backup ownership
// is resolved here; selection and rendering consume its immutable value result.
static captured_depth capture_record(effect_runtime *runtime, const generic_depth_data &data,
	const generic_depth_device_data &device_data, resource source, uint64_t identity, resource_view view,
	bool using_backup, uint64_t capture_after = 0, bool presentation = true)
{
	captured_depth result;
	std::shared_lock<std::shared_mutex> lock(s_mutex);
	const auto found = device_data.depth_stencil_resources.find(source);
	if (found == device_data.depth_stencil_resources.end() || found->second.identity != identity) return result;
	const auto &info = found->second;
	resource sampled = source;
	uint64_t assignment = identity;
	bool published = true;
	if (using_backup)
	{
		const auto backup = std::find_if(device_data.depth_stencil_backups.begin(), device_data.depth_stencil_backups.end(),
			[source](const auto &entry) { return entry.depth_stencil == source; });
		if (backup == device_data.depth_stencil_backups.end()) return result;
		sampled = backup->backup_texture; assignment = backup->content_identity;
		if (runtime->get_device()->get_api() == device_api::d3d12)
		{
			const auto &ticket = info.last_frame_stats.capture_ticket;
			published = ticket && backup->published_ticket == ticket.id &&
				backup->published_present == device_data.native_present_index;
			if (!presentation)
			{
				sampled = {ticket.texture}; assignment = ticket.resource_id;
				published = bool(ticket);
			}
		}
	}
	const auto region = effective_capture_region(info);
	auto &out = result.record;
	out.source = source.handle; out.identity = identity; out.sampled = sampled.handle; out.assignment = assignment; out.view = view.handle;
	out.runtime = data.runtime_epoch; out.present = device_data.native_depth_present_index; out.frame = info.last_used_in_frame;
	out.layout = info.layout_epoch; out.capture_after = capture_after;
	out.shape = {info.desc.texture.width, info.desc.texture.height, static_cast<uint32_t>(info.desc.texture.format),
		region.x, region.y, region.width, region.height};
	out.active = info.last_frame_stats.total.has_work();
	out.preserved = info.last_frame_stats.copied_during_frame && published;
	out.ambiguous = info.last_frame_stats.copy_provenance_ambiguous;
	out.direct = !using_backup;
	result.end_of_frame = info.last_frame_stats.end_of_frame_copy;
	auto &depth = result.metadata;
	depth.source_resource = source; depth.resource = sampled; depth.shader_resource = view;
	depth.width = out.shape.width; depth.height = out.shape.height;
	depth.x = region.x; depth.y = region.y; depth.active_width = region.width; depth.active_height = region.height;
	depth.source_id = identity; depth.backup_id = using_backup ? assignment : 0;
	depth.layout_epoch = info.layout_epoch; depth.frame_index = info.last_used_in_frame; depth.runtime_epoch = data.runtime_epoch;
	depth.detected_orientation = info.orientation_evidence.detected();
	depth.orientation_agreeing_frames = info.orientation_evidence.agreeing_frames();
	depth.ready = out.current(device_data.native_present_index, device_data.frame_index, data.runtime_epoch);
	lock.unlock();
	if (view != 0 && runtime->get_device()->get_resource_from_view(view) != sampled) return {};
	return result;
}

SUNSHINE_DEPTH_ADDON_INTERNAL bool validate_capture_record(effect_runtime *runtime, const generic_depth_data &data,
	const generic_depth_device_data &device_data, const sunshine_depth_capture::capture_record &record)
{
	const auto current = capture_record(runtime, data, device_data, resource{record.source}, record.identity,
		resource_view{record.view}, !record.direct, record.capture_after).record;
	return current.current(device_data.native_present_index, device_data.frame_index, data.runtime_epoch) &&
		current.sampled == record.sampled && current.assignment == record.assignment && current.layout == record.layout &&
		current.shape == record.shape && current.frame == record.frame && current.present == record.present;
}

SUNSHINE_DEPTH_ADDON_INTERNAL hold_gate judge_hold_gate(const depth_selection_snapshot &snapshot,
	const sunshine_depth_capture::capture_set &captures, uint64_t held, uint64_t frame,
	uint32_t color_width, uint32_t color_height, bool resolve_depth, uint64_t grace)
{
	return sunshine_game3d::depth_output::judge_hold_gate(snapshot, captures, held, frame,
		color_width, color_height, resolve_depth, grace,
		[](const auto &info, uint32_t width, uint32_t height) {
			return candidate_matches_shape(info, width, height);
		});
}

// Whether this Present may show the capture shown on the runtime's previous
// Present again, as reused depth, when no current copy can be shown: the owner's
// hold of the last good depth for at most the source age (freshness.h), the
// same rule as the API providers' missing-frame hold. The capture must be the
// one data.shown names (so older depth never follows newer), still bound
// through the same view of the same backup, which must still hold exactly the
// copy it showed (its content serial; any later copy into it ends the hold),
// of the same source lifetime, layout and crop, for the same runtime epoch,
// anchor and colour size, within the source age of the Present on which it was
// current, on a Present whose hold gates are open (judge_hold_gate). A hold
// never advances the frame, calibration or coverage records.
// check_age is false for get_frame_depth's revalidation within the same Present.
static bool hold_shown_capture(effect_runtime *runtime, const generic_depth_data &data,
	const generic_depth_device_data &device_data, uint32_t color_width, uint32_t color_height, bool check_age = true)
{
	const auto &shown = data.shown;
	const auto &record = shown.record;
	if (!sunshine_game3d::depth_output::generic_capture_matches(data, color_width, color_height,
		check_age, [] { return GetTickCount64(); })) return false;
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		const auto found = device_data.depth_stencil_resources.find(resource{record.source});
		if (found == device_data.depth_stencil_resources.end()) return false;
		const auto &info = found->second;
		const auto region = effective_capture_region(info);
		if (info.identity != record.identity || info.layout_epoch != record.layout ||
			info.desc.texture.width != record.shape.width || info.desc.texture.height != record.shape.height ||
			static_cast<uint32_t>(info.desc.texture.format) != record.shape.format || region.x != record.shape.x ||
			region.y != record.shape.y || region.width != record.shape.extent_width || region.height != record.shape.extent_height) return false;
		bool intact = false;
		for (const auto &backup : device_data.depth_stencil_backups)
			if (backup.depth_stencil.handle == record.source)
				intact = backup.references && backup.backup_texture.handle == record.sampled &&
					backup.content_identity == record.assignment && backup.content_serial == shown.content_serial;
		if (!intact) return false;
	}
	return runtime->get_device()->get_resource_from_view(resource_view{record.view}).handle == record.sampled;
}

// The capture budget's view of the snapshot: geometry and content qualification.
static std::vector<sunshine_depth_capture::source_observation> capture_observations(const generic_depth_data &data,
	const depth_selection_snapshot &snapshot)
{
	std::vector<sunshine_depth_capture::source_observation> observations;
	observations.reserve(snapshot.size());
	for (const auto &[value, info] : snapshot)
	{
		if (!content_supported(info.desc) || info.last_used_in_frame == std::numeric_limits<uint64_t>::max()) continue;
		const auto region = effective_capture_region(info);
		const auto *e = data.selection.evidence(info.identity);
		observations.push_back({info.identity, info.layout_epoch, info.last_used_in_frame,
			{info.desc.texture.width, info.desc.texture.height, static_cast<uint32_t>(info.desc.texture.format),
			 region.x, region.y, region.width, region.height},
			e && e->confirmed_good && e->quality.kind == sunshine_depth::content_kind::useful});
	}
	return observations;
}

// Present coverage for content selection, after this Present's routing: what
// the anchor's capture set did and whether it gave a ready current copy, and
// what each candidate's prospective capture set did. Work that makes an
// end-of-frame copy current (raster, mesh, or a transfer that is depth of its
// own, resolve_transfer_lineage) decides a miss; only raster and mesh work
// cover another source's gap (docs/reshade-sbs.md, Selecting scene depth).
// Both survive a merge with counters off: work through observed_work, a draw
// conservatively not at all.
static void observe_present_coverage(generic_depth_data &data, uint64_t frame, const depth_selection_snapshot &snapshot)
{
	std::vector<uint64_t> active, drew;
	for (const auto &[value, info] : snapshot)
	{
		if (info.last_used_in_frame != frame) continue;
		if (info.last_frame_stats.total.has_work()) active.push_back(info.identity);
		if (info.last_frame_stats.total.drew()) drew.push_back(info.identity);
	}
	const auto contains = [](const std::vector<uint64_t> &ids, uint64_t id) { return std::find(ids.begin(), ids.end(), id) != ids.end(); };
	sunshine_depth::selection_policy::set_presence anchor;
	for (unsigned i = 0; i != data.captures.count; ++i)
	{
		anchor.active = anchor.active || contains(active, data.captures.records[i].identity);
		anchor.drew = anchor.drew || contains(drew, data.captures.records[i].identity);
	}
	const auto observations = capture_observations(data, snapshot);
	data.selection.observe_present(frame, data.anchor_identity, data.capture_ready, anchor, [&](uint64_t id) {
		return sunshine_depth::selection_policy::set_presence {
			sunshine_depth_capture::prospective_set_contains(id, observations, active),
			sunshine_depth_capture::prospective_set_contains(id, observations, drew)};
	});
}

// A stable shader/probe surface receives an admitted snapshot. It never owns
// end-of-frame pixels: those resources and read leases belong to capture owner.
static void publish_preserved_depth(effect_runtime *runtime, command_list *commands,
	generic_depth_device_data &device_data, resource source, uint64_t identity, uint64_t capture_after)
{
	auto &data = *runtime->get_private_data<generic_depth_data>();
	const auto frame = device_data.frame_index, present = device_data.native_present_index;
	const auto current = capture_record(runtime, data, device_data, source, identity, {}, true, capture_after, false);
	if (!current.record.current(present, frame, data.runtime_epoch)) return;
	sunshine_streamline::depth_capture::preservation_ticket ticket;
	resource destination{};
	uint64_t assignment{};
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		const auto found = device_data.depth_stencil_resources.find(source);
		const auto *view = device_data.find_depth_stencil_backup(source);
		if (found == device_data.depth_stencil_resources.end() || found->second.identity != identity || !view || !view->references ||
			device_data.frame_index != frame || device_data.native_present_index != present) return;
		ticket = found->second.last_frame_stats.capture_ticket;
		if (!ticket || ticket.texture != current.record.sampled || ticket.resource_id != current.record.assignment ||
			(view->published_ticket == ticket.id && view->published_present == present)) return;
		destination = view->backup_texture; assignment = view->content_identity;
	}
	if (!sunshine_streamline::depth_capture::copy_preserved(commands->get_native(), runtime->get_command_queue()->get_native(),
		ticket, destination.handle, D3D12_RESOURCE_STATE_COPY_DEST)) return;
	const std::unique_lock<std::shared_mutex> lock(s_mutex);
	for (auto &backup : device_data.depth_stencil_backups)
		if (backup.backup_texture == destination) backup.content_serial = ++device_data.next_content_serial;
	const auto found = device_data.depth_stencil_resources.find(source);
	auto *view = device_data.find_depth_stencil_backup(source);
	if (found == device_data.depth_stencil_resources.end() || found->second.identity != identity || !view ||
		view->content_identity != assignment || found->second.last_frame_stats.capture_ticket.id != ticket.id ||
		device_data.frame_index != frame || device_data.native_present_index != present) return;
	view->published_ticket = ticket.id;
	view->published_present = present;
}

static void preserve_d3d12_at_effects(effect_runtime *runtime, command_list *commands,
	generic_depth_device_data &device_data, resource source, uint64_t identity, uint64_t capture_after)
{
	const auto frame = device_data.frame_index, present = device_data.native_present_index;
	viewport copied_viewport{};
	bool capture = false;
	uint64_t reservation = 0;
	{
		const std::unique_lock<std::shared_mutex> lock(s_mutex);
		const auto found = device_data.depth_stencil_resources.find(source);
		const auto *view = device_data.find_depth_stencil_backup(source);
		if (found == device_data.depth_stencil_resources.end() || found->second.identity != identity || !view || !view->references ||
			device_data.native_depth_present_index != present || !frame || (capture_after && frame <= capture_after)) return;
		auto &info = found->second;
		if (info.last_used_in_frame != frame || !info.last_frame_stats.total.has_work() || info.last_frame_stats.copy_provenance_ambiguous) return;
		capture = !info.last_frame_stats.copied_during_frame &&
			!info.last_frame_stats.preservation_reservation && (info.desc.usage & resource_usage::copy_source) != 0;
		if (capture) {
			static std::atomic<uint64_t> next_reservation{1};
			reservation = next_reservation.fetch_add(1, std::memory_order_relaxed);
			info.last_frame_stats.preservation_reservation = reservation;
			info.last_frame_stats.copy_provenance_ambiguous = true;
		}
		copied_viewport = info.last_frame_stats.total.last_viewport;
	}
	if (capture)
	{
		// Nothing names the source's state at the end of the frame: only the
		// state the game's executed work proved on this queue (or a state
		// ReShade's own list observed) admits the copy; otherwise the earlier
		// capture holds.
		sunshine_streamline::depth_capture::record_diagnostic refusal;
		auto ticket = sunshine_streamline::depth_capture::record_preserved(commands->get_native(), source.handle,
			sunshine_api_contract::generic::end_of_frame, runtime->get_command_queue()->get_native(), &refusal);
		auto *data = runtime->get_private_data<generic_depth_data>();
		if (!ticket && data)
		{
			++data->end_of_frame_refused;
			data->end_of_frame_refusal = refusal.stage;
		}
		{
			const std::unique_lock<std::shared_mutex> lock(s_mutex);
			const auto found = device_data.depth_stencil_resources.find(source);
			if (found == device_data.depth_stencil_resources.end() || found->second.identity != identity ||
				device_data.frame_index != frame || device_data.native_present_index != present ||
				found->second.last_frame_stats.preservation_reservation != reservation) return;
			auto &stats = found->second.last_frame_stats;
			stats.copy_provenance_ambiguous = false;
			if (!ticket) {
				stats.preservation_reservation = 0;
				return;
			}
			stats.capture_ticket = std::move(ticket);
			stats.copied_during_frame = true; stats.copied_viewport = copied_viewport;
			stats.end_of_frame_copy = true;
			observe_preserved_layout(found->second, stats);
		}
	}
	publish_preserved_depth(runtime, commands, device_data, source, identity, capture_after);
}

// Produce an allowed end-of-frame copy before consumers decide which current
// source can render. This is Generic depth's only copy point.
// Probe, capture-member and legacy selected paths share the same live facts;
// a selection snapshot cannot authorize a second copy or an inactive present.
static void preserve_depth_at_effects(effect_runtime *runtime, command_list *cmd_list,
	generic_depth_device_data &device_data, resource source, uint64_t identity, uint64_t capture_after = 0)
{
	device *const device = runtime->get_device();
	const auto api = device->get_api();
	if (api == device_api::d3d12)
	{
		// ReShade's own list records this copy; its list events never report it.
		sunshine_streamline::depth_capture::observe_runtime_list(
			runtime->get_command_queue()->get_immediate_command_list()->get_native(), runtime->get_command_queue()->get_native());
		preserve_d3d12_at_effects(runtime, cmd_list, device_data, source, identity, capture_after);
		return;
	}
	resource_desc desc;
	resource backup_texture {};
	uint64_t assignment = 0;
	viewport copied_viewport {};
	const uint64_t frame = device_data.frame_index, present = device_data.native_present_index;
	{
		const std::unique_lock<std::shared_mutex> lock(s_mutex);
		const auto found = device_data.depth_stencil_resources.find(source);
		const auto *backup = device_data.find_depth_stencil_backup(source);
		if (found == device_data.depth_stencil_resources.end() || found->second.identity != identity ||
			!backup || !backup->references || backup->backup_texture == 0 ||
			device_data.native_depth_present_index != present || !frame || (capture_after && frame <= capture_after)) return;
		const auto &info = found->second;
		if (info.last_used_in_frame != frame || !info.last_frame_stats.total.has_work() ||
			info.last_frame_stats.copied_during_frame || info.last_frame_stats.copy_provenance_ambiguous ||
			(info.desc.usage & (resource_usage::copy_source | resource_usage::resolve_source)) == 0) return;
		// The copy holds what the source holds now: a game that cleared or
		// reused it after its scene leaves flat depth here, and no earlier
		// copy point exists (on D3D11/12 the content checks refuse it).
		desc = info.desc;
		backup_texture = backup->backup_texture; assignment = backup->content_identity;
		copied_viewport = info.last_frame_stats.total.last_viewport;
		// Reserve this source/frame before issuing GPU commands. Another runtime
		// must neither write the same backup concurrently nor sample it mid-copy.
		found->second.last_frame_stats.copied_during_frame = true;
		found->second.last_frame_stats.copy_provenance_ambiguous = true;
	}
	const resource_usage old_state = desc.usage & (resource_usage::depth_stencil | resource_usage::shader_resource);
	if (desc.texture.samples > 1)
	{
		if (!device->check_capability(device_caps::resolve_depth_stencil)) return;
		cmd_list->barrier(source, old_state, resource_usage::resolve_source);
		cmd_list->barrier(backup_texture, resource_usage::copy_dest, resource_usage::resolve_dest);
		cmd_list->resolve_texture_region(source, 0, nullptr, backup_texture, 0, 0, 0, 0, format_to_default_typed(desc.texture.format));
		cmd_list->barrier(backup_texture, resource_usage::resolve_dest, resource_usage::copy_dest);
		cmd_list->barrier(source, resource_usage::resolve_source, old_state);
	}
	else
	{
		cmd_list->barrier(source, old_state, resource_usage::copy_source);
		cmd_list->copy_resource(source, backup_texture);
		cmd_list->barrier(source, resource_usage::copy_source, old_state);
	}
	const std::unique_lock<std::shared_mutex> lock(s_mutex);
	for (auto &entry : device_data.depth_stencil_backups)
		if (entry.backup_texture == backup_texture) entry.content_serial = ++device_data.next_content_serial;
	const auto found = device_data.depth_stencil_resources.find(source);
	const auto *backup = device_data.find_depth_stencil_backup(source);
	if (found == device_data.depth_stencil_resources.end() || found->second.identity != identity ||
		!backup || backup->backup_texture != backup_texture || backup->content_identity != assignment || !backup->references ||
		device_data.frame_index != frame || device_data.native_present_index != present ||
		device_data.native_depth_present_index != present || found->second.last_used_in_frame != frame) return;
	auto &stats = found->second.last_frame_stats;
	stats.copied_during_frame = true;
	stats.copied_viewport = copied_viewport;
	stats.copy_provenance_ambiguous = false;
	stats.end_of_frame_copy = true;
	observe_preserved_layout(found->second, stats);
}

static generic_depth_data::member_view *select_captured_depth(effect_runtime *runtime, command_list *cmd_list, generic_depth_data &data,
	generic_depth_device_data &device_data, const depth_selection_snapshot &snapshot, bool automatic,
	resource &selected, const depth_selection_resource *&selected_info)
{
	data.capture_wait_detail = nullptr;
	data.hold_only = false;
	retire_member_views(runtime, data, device_data);
	const auto observations = capture_observations(data, snapshot);
	const uint64_t anchor = selected_info ? selected_info->identity : 0;
	const auto group = data.capture_budget.update(anchor, observations, !automatic);
	data.anchor_depth_stencil = selected;
	data.anchor_identity = anchor;
	data.rotating = group.count > 1;
	for (auto &member : data.member_views)
	{
		if (member.view == 0) continue;
		const bool retained = std::find(group.members.begin(), group.members.begin() + group.count, member.identity) != group.members.begin() + group.count;
		if (retained) member.retire_after = 0;
		else if (!member.retire_after) member.retire_after = device_data.frame_index + 50;
	}
	// Capture allocation precedes qualification/current selection. Otherwise a
	// new source could never produce the evidence needed to become selectable.
	if (automatic || data.raw_scene_requested)
		for (unsigned i = 0; i != group.count; ++i)
		{
			if (std::any_of(data.member_views.begin(), data.member_views.end(), [&](const auto &entry) {
				return entry.identity == group.members[i] && entry.view != 0;
			})) continue;
			auto slot = std::find_if(data.member_views.begin(), data.member_views.end(), [](const auto &entry) { return entry.view == 0; });
			if (slot == data.member_views.end()) break; // Bounded retirement; never wait or allocate a ninth cached view.
			const auto source = std::find_if(snapshot.begin(), snapshot.end(), [&](const auto &entry) { return entry.second.identity == group.members[i]; });
			if (source == snapshot.end()) continue;
			if (data.selected_identity == source->second.identity && data.selected_shader_resource != 0 && data.using_backup_texture)
			{
				*slot = {source->first, source->second.identity, 0, data.selected_shader_resource}; // Transfer existing ownership.
				continue;
			}
			const bool already_tracked = device_data.find_depth_stencil_backup(source->first) != nullptr;
			auto *backup = device_data.track_depth_stencil_for_backup(runtime->get_device(), source->first, source->second.desc);
			if (!backup) continue;
			resource_view view {};
			const resource_view_desc desc(format_to_default_typed(source->second.desc.texture.format));
			if (!runtime->get_device()->create_resource_view(backup->backup_texture, resource_usage::shader_resource, desc, &view))
			{
				device_data.untrack_depth_stencil(runtime->get_device(), source->first);
				continue;
			}
			// The frame's copied flag can describe a retired probe assignment.
			// Newly assigned storage needs a later legal copy of its own.
			*slot = {source->first, source->second.identity, 0, view, already_tracked ? 0 : device_data.frame_index};
		}
	// End-of-frame capture cannot depend on already having a selected current
	// copy.
	for (const auto &member : data.member_views)
		if (!member.retire_after && member.view != 0)
			preserve_depth_at_effects(runtime, cmd_list, device_data, member.source, member.identity, member.capture_after);
	sunshine_depth::raw_source_roster next_roster;
	next_roster.routing_epoch = group.epoch;
	next_roster.basis_epoch = data.raw_basis_epoch;
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		for (unsigned i = 0; i != group.count; ++i)
			for (const auto &[value, info] : device_data.depth_stencil_resources)
				if (info.identity == group.members[i])
				{
					const auto region = effective_capture_region(info);
					auto &out = next_roster.members[next_roster.count++];
					out.source = {value.handle, info.identity, 0, info.desc.texture.width, info.desc.texture.height,
						region.x, region.y, region.width, region.height};
					out.layout_epoch = info.layout_epoch;
					out.direction = info.orientation_evidence.detected();
					break;
				}
	}
	data.raw_roster = next_roster;
	for (auto &entry : data.raw_members)
	{
		if (!entry.identity) continue;
		const auto found = std::find_if(data.raw_roster.members.begin(), data.raw_roster.members.begin() + data.raw_roster.count,
			[&entry](const auto &member) { return member.source.lifetime == entry.identity; });
		if (found == data.raw_roster.members.begin() + data.raw_roster.count) entry = {};
		else if (entry.available && (!(entry.sample.metadata.source == found->source) ||
			entry.sample.metadata.layout_epoch != found->layout_epoch || entry.sample.metadata.direction != found->direction))
		{
			const auto id = entry.identity;
			entry = {}; entry.identity = id;
		}
	}
	for (unsigned i = 0; i != data.raw_roster.count; ++i) raw_member(data, data.raw_roster.members[i].source.lifetime, true);
	data.captures = {};
	data.captures.present = device_data.native_present_index;
	data.captures.frame = device_data.frame_index;
	data.captures.runtime = data.runtime_epoch;
	sunshine_depth_capture::preference preferred;
	preferred.anchor = anchor; preferred.single_anchor = !automatic;
	for (unsigned i = 0; i != group.count; ++i)
	{
		const auto observed = std::find_if(observations.begin(), observations.end(), [&](const auto &o) { return o.id == group.members[i]; });
		if (observed != observations.end() && observed->qualified) preferred.qualified[preferred.count++] = observed->id;
		const auto source = std::find_if(snapshot.begin(), snapshot.end(), [&](const auto &entry) { return entry.second.identity == group.members[i]; });
		if (source == snapshot.end()) continue;
		resource_view view {};
		uint64_t floor = 0;
		bool backup = true;
		for (const auto &entry : data.member_views)
			if (entry.identity == group.members[i] && entry.view != 0) { view = entry.view; floor = entry.capture_after; break; }
		if (view == 0 && data.selected_identity == group.members[i]) { view = data.selected_shader_resource; backup = data.using_backup_texture; }
		const auto captured = capture_record(runtime, data, device_data, source->first, source->second.identity, view, backup, floor);
		data.captures.records[data.captures.count] = captured.record;
		++data.captures.count;
	}
	// Same-geometry peers retain stable ties, while the selector's established
	// quality score orders fallback choices. A usable anchor always wins first.
	std::stable_sort(preferred.qualified.begin(), preferred.qualified.begin() + preferred.count,
		[&data](uint64_t a, uint64_t b) { return data.selection.retention_score(a) > data.selection.retention_score(b); });
	const auto decision = sunshine_depth_capture::choose_current(data.captures, preferred);
	// Discovery can fill uncovered alternating presents before the full lifetime
	// round robin returns here. The hint changes only the next probe visit: the
	// candidate still needs its own preserved copies and ordinary content checks.
	uint64_t discovery_id = 0, discovery_layout = 0, discovery_draws = 0;
	if (automatic && data.raw_scene_requested &&
		decision.reason == sunshine_depth_capture::selection_status::no_current_copy)
	{
		const auto primary = std::find_if(observations.begin(), observations.end(),
			[anchor](const auto &o) { return o.id == anchor; });
		if (primary != observations.end())
			for (const auto &[value, info] : snapshot)
			{
				const auto *evidence = data.selection.evidence(info.identity);
				const auto region = effective_capture_region(info);
				const sunshine_depth_capture::geometry shape {info.desc.texture.width, info.desc.texture.height,
					static_cast<uint32_t>(info.desc.texture.format), region.x, region.y, region.width, region.height};
				if (!evidence || (evidence->confirmed_good && evidence->quality.kind == sunshine_depth::content_kind::useful) ||
					!(shape == primary->shape) || info.last_used_in_frame != device_data.frame_index ||
					info.last_frame_stats.total.drawcalls == 0 ||
					std::find(group.members.begin(), group.members.begin() + group.count, info.identity) != group.members.begin() + group.count)
					continue;
				const uint64_t draws = info.last_frame_stats.total.drawcalls;
				if (!discovery_id || draws > discovery_draws || (draws == discovery_draws && info.identity < discovery_id))
				{
					discovery_id = info.identity; discovery_layout = info.layout_epoch; discovery_draws = draws;
				}
			}
	}
	data.discovery_hint.observe(anchor, device_data.frame_index, discovery_id, discovery_layout);
#include "generic_trace.inl"
	if (data.raw_scene_requested || (automatic && group.count > 1))
	{
		const depth_selection_resource *current_info = nullptr;
		resource current_source {};
		// Without a current copy, the capture shown on the previous Present stays
		// selected so it can be held (hold_shown_capture decides after this
		// Present's copies); otherwise nothing is bound.
		const auto &shown = data.shown;
		const auto current_id = decision.source ? decision.source :
			shown.record.identity && shown.anchor == anchor && shown.record.runtime == data.runtime_epoch &&
			sunshine_freshness::within_source_age(shown.tick, GetTickCount64()) ? shown.record.identity : 0;
		for (const auto &[value, info] : snapshot)
			if (current_id && info.identity == current_id &&
				std::any_of(data.member_views.begin(), data.member_views.end(), [&](const auto &entry) {
					return entry.identity == info.identity && entry.view != 0 &&
						(!entry.capture_after || device_data.frame_index > entry.capture_after);
				}))
			{
				current_info = &info; current_source = value; break;
			}
		// Different sources may legitimately render together. Only the chosen
		// source's current-copy facts and independent preference decide binding.
		selected = current_source;
		selected_info = current_info;
		data.hold_only = current_info && !decision.source;
		if (!current_info)
			data.capture_wait_detail = "Waiting for a qualified current depth capture.";
		if (s_frame_activity_trace)
		{
			auto &trace = data.trace_capture;
			trace.reason = current_info ? sunshine_depth_capture::name(decision.reason) : !current_id ?
				sunshine_depth_capture::name(decision.reason) :
				(trace.raster & trace.floors) ? "new_assignment_floor" : "missing_cached_view";
		}
	}
	for (auto &entry : data.member_views)
		if (!entry.retire_after && entry.view != 0 && selected_info && entry.identity == selected_info->identity)
		{
			if (entry.capture_after && device_data.frame_index <= entry.capture_after)
			{
				selected = {}; selected_info = nullptr;
				data.capture_wait_detail = "The depth backup is newly assigned; waiting for its first current capture.";
				if (s_frame_activity_trace) data.trace_capture.reason = "new_assignment_floor";
				return nullptr; // Repeated selection or effects changes cannot bypass the new-assignment floor.
			}
			return &entry;
		}
	return nullptr;
}

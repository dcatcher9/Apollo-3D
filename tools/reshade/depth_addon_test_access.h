/*
 * Copyright (C) 2021 Patrick Mours
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Sunshine changes: bounded content-based scene depth selection and upscaler
 * eligibility, based on ReShade 6.8.0 example 09-depth, commit
 * 18deaa52de0c425a78b329e9cb3c497281cd00ec. The original manual depth controls
 * and capture paths remain available inside the Sunshine 3D add-on. This module
 * never reads another add-on's private C++ objects.
 */

// depth_addon.cpp's shared state and depth-resource types, and in the test
// add-on the internals its exports (test_depth_addon_exports.cpp) reach. Only
// those two files include this header.
//
// The shipping add-on keeps every one of them internal (static) to
// depth_addon.cpp. The test add-on (SUNSHINE_SBS_RUNTIME_TEST_ADDON) makes the
// definitions below inline and the later ones external, so that its exports
// can run in their own translation unit of the same DLL.
#pragma once

#include "addon_lifetime.h"
#include <Windows.h>
#include <TlHelp32.h>
#include <Unknwn.h>
#include <d3d12.h>
#include <imgui.h>
#include <reshade.hpp>
#include "async_log.h"
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <unordered_map>
#include <cmath> // std::abs, std::modf
#include <cstdio> // std::snprintf
#include <cstring> // std::strcmp
#include <algorithm> // std::find_if, std::remove, std::sort
#include <atomic>
#include <array>
#include <optional>
#include <limits>
#include <string>
#include <type_traits>
#include "upscaler_call_trace.h"
#include "depth_addon.h"
#include "depth_content_sampler.h"
#include "depth_selection_policy.h"
#include "depth_capture_policy.h"
#include "depth_probe_policy.h"
#include "game3d_controls.h"
#include "diagnostic_log_gate.h"
#include "streamline_camera_probe.h"
#include "streamline_depth_capture.h"
#include "streamline_depth_provider.h"
#include "native_resource_identity.h"
#include "camera_sample_observation.h"
#include "game3d_slow_step.h"

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
#define SUNSHINE_DEPTH_ADDON_SHARED inline
#define SUNSHINE_DEPTH_ADDON_INTERNAL
#else
#define SUNSHINE_DEPTH_ADDON_SHARED static
#define SUNSHINE_DEPTH_ADDON_INTERNAL static
#endif

using namespace reshade::api;

SUNSHINE_DEPTH_ADDON_SHARED std::shared_mutex s_mutex;
SUNSHINE_DEPTH_ADDON_SHARED bool s_sunshine_auto = true;
SUNSHINE_DEPTH_ADDON_SHARED bool s_registered = false;
SUNSHINE_DEPTH_ADDON_SHARED bool s_initialized = false;
// Fixed for each add-on registration; changing the diagnostic requires restart.
SUNSHINE_DEPTH_ADDON_SHARED bool s_streamline_probe_events = false;
SUNSHINE_DEPTH_ADDON_SHARED bool s_streamline_source_events = false;
SUNSHINE_DEPTH_ADDON_SHARED bool s_frame_activity_trace = false; // Opt-in metadata only; never changes capture policy.
SUNSHINE_DEPTH_ADDON_SHARED const char *s_status_message = "Depth selection has not been initialized.";
SUNSHINE_DEPTH_ADDON_SHARED constexpr const char *s_overlay_title = "Sunshine 3D";
SUNSHINE_DEPTH_ADDON_SHARED std::atomic<uint64_t> s_next_command_identity { 1 };
SUNSHINE_DEPTH_ADDON_SHARED std::atomic<uint64_t> s_next_sample_scope { 1 };

SUNSHINE_DEPTH_ADDON_SHARED bool content_supported(const resource_desc &desc)
{
	return desc.type == resource_type::texture_2d && desc.texture.levels == 1 &&
		desc.texture.depth_or_layers == 1 && desc.texture.samples == 1;
}

// One challenger may be tracked across all runtimes/devices. The sampler also
// permits only one globally in-flight GPU readback. Ownership is short-lived
// and no native resource address is ever saved in configuration.
SUNSHINE_DEPTH_ADDON_SHARED effect_runtime *s_challenger_owner = nullptr;
SUNSHINE_DEPTH_ADDON_SHARED std::mutex s_challenger_mutex;

enum class draw_stats_heuristic : unsigned int
{
	prefer_vertices = 0,
	vertices,
	drawcalls
};
enum class aspect_ratio_heuristic : unsigned int
{
	none = 0,
	similar_aspect_ratio,
	multiples_of_resolution,
	match_resolution_exactly,
	match_custom_resolution_exactly
};

SUNSHINE_DEPTH_ADDON_SHARED bool s_disable_intz = false;
// Enable or disable the creation of backup copies at clear operations on the selected depth-stencil
SUNSHINE_DEPTH_ADDON_SHARED unsigned int s_preserve_depth_buffers = 0;
// Choose impact of draw statistics in the detection heuristic
SUNSHINE_DEPTH_ADDON_SHARED draw_stats_heuristic s_draw_stats_heuristic = draw_stats_heuristic::prefer_vertices;
// Enable or disable the aspect ratio check from 'check_aspect_ratio' in the detection heuristic
SUNSHINE_DEPTH_ADDON_SHARED aspect_ratio_heuristic s_aspect_ratio_heuristic = aspect_ratio_heuristic::similar_aspect_ratio;
// Enable or disable the format check from 'check_depth_format' in the detection heuristic
SUNSHINE_DEPTH_ADDON_SHARED unsigned int s_format_filtering = 0;
SUNSHINE_DEPTH_ADDON_SHARED unsigned int s_custom_resolution_filtering[2] = {};

enum class clear_op : uint8_t
{
	clear_depth_stencil_view,
	fullscreen_draw,
	unbind_depth_stencil_view,
	direct_depth_switch,
};

struct depth_tracking_demand
{
	bool counters = true;
	bool clear_history = true;
};

struct draw_stats
{
	uint32_t vertices = 0;
	uint32_t drawcalls = 0;
	uint32_t drawcalls_indirect = 0;
	uint32_t scene_writes = 0; // Mesh dispatches and depth copies/resolves have no known vertex count.
	viewport last_viewport = {};
	bool observed_work = false; // Preservation eligibility does not require candidate counters.

	bool has_work() const { return observed_work || drawcalls != 0 || scene_writes != 0; }
	uint64_t work_count() const { return uint64_t(drawcalls) + scene_writes; }
	void clear_counters()
	{
		observed_work = has_work();
		vertices = drawcalls = drawcalls_indirect = scene_writes = 0;
	}

	bool operator>(const draw_stats &other) const
	{
		if (s_draw_stats_heuristic == draw_stats_heuristic::vertices)
			return vertices > other.vertices;
		if (s_draw_stats_heuristic == draw_stats_heuristic::drawcalls)
			return work_count() > other.work_count();

		return (scene_writes == 0 && drawcalls_indirect < (drawcalls / 3) ?
			// Choose snapshot with the most vertices, since that is likely to contain the main scene
			vertices > other.vertices :
			// Or check draw calls, since vertices may not be accurate if application is using indirect draw calls
			work_count() > other.work_count());
	}
};
struct clear_stats : public draw_stats
{
	clear_op clear_op = clear_op::clear_depth_stencil_view;
	bool copied_during_frame = false;
};

struct depth_stencil_frame_stats
{
	draw_stats total;
	draw_stats current; // Stats since last clear operation
	// Command-list-local proof only. A real D3D12 depth clear requires DEPTH_WRITE;
	// any later barrier or opaque execution revokes this rather than inferring
	// native state from ReShade's lossy resource_usage conversion.
	bool direct_clear_depth_write = false;
	draw_stats best_copy_stats; // Per resource, so a busy decoy cannot starve a low-draw scene.
	viewport copied_viewport = {}; // Last submitted copy, independent of greatest workload.
	bool copy_provenance_ambiguous = false;
	sunshine_streamline::commands::recording_marker capture_marker {};
	sunshine_streamline::content::copy_snapshot depth_copy {};
	std::vector<clear_stats> clears;
	bool copied_during_frame = false;
	// Pixel storage and GPU lifetime belong to the common capture owner. This
	// lease survives recording reset until the submitted frame is consumed.
	sunshine_streamline::depth_capture::preservation_ticket capture_ticket;
	// Only this reservation may publish an EOF capture for the source/frame.
	uint64_t preservation_reservation = 0;
	bool reversed_clear_value = false;
	bool clear_zero = false, clear_one = false, clear_other = false;
	// Passive transport capability: a real scene was followed by a boundary
	// where this preservation mode can copy. This is never pixel-readiness proof.
	bool preservation_boundary = false;

	void limit_tracking(depth_tracking_demand demand)
	{
		if (!demand.counters)
		{
			total.clear_counters();
			current.clear_counters();
			best_copy_stats.clear_counters();
		}
		if (!demand.clear_history) clears.clear();
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
	bool inside_render_pass = false;
	std::unordered_map<resource, depth_stencil_frame_stats, resource_hash> stats_per_used_depth_stencil;
	bool first_draw_since_bind = true;
	explicit state_tracking(bool is_queue) : is_queue(is_queue)
	{
		// Reserve some space upfront to avoid rehashing during command recording
		stats_per_used_depth_stencil.reserve(32);
	}

	void reset()
	{
		stats_per_used_depth_stencil.clear();
		current_depth_stencil = { 0 };
		current_depth_lifetime = 0;
		render_pass_depth = {};
		render_pass_depth_uncertain = false;
		inside_render_pass = false;
	}
	void invalidate_direct_clear_proofs()
	{
		for (auto &[depth, stats] : stats_per_used_depth_stencil)
			stats.direct_clear_depth_write = false;
	}
	void reset_on_present()
	{
		assert(is_queue);
		stats_per_used_depth_stencil.clear();
	}

	void merge(const state_tracking &source, depth_tracking_demand demand = {})
	{
		// Executing a command list in a different command list inherits state
		current_depth_stencil = source.current_depth_stencil;
		current_depth_lifetime = source.current_depth_lifetime;

		if (source.stats_per_used_depth_stencil.empty())
			return;

		stats_per_used_depth_stencil.reserve(source.stats_per_used_depth_stencil.size());
		for (const auto &[depth_stencil, source_stats] : source.stats_per_used_depth_stencil)
		{
			depth_stencil_frame_stats &stats = stats_per_used_depth_stencil[depth_stencil];
			stats.limit_tracking(demand);
			stats.total.observed_work |= source_stats.total.has_work();
			stats.current.observed_work |= source_stats.current.has_work();
			if (demand.counters)
			{
				stats.total.vertices += source_stats.total.vertices;
				stats.total.drawcalls += source_stats.total.drawcalls;
				stats.total.drawcalls_indirect += source_stats.total.drawcalls_indirect;
				stats.total.scene_writes += source_stats.total.scene_writes;
				stats.current.vertices += source_stats.current.vertices;
				stats.current.drawcalls += source_stats.current.drawcalls;
				stats.current.drawcalls_indirect += source_stats.current.drawcalls_indirect;
				stats.current.scene_writes += source_stats.current.scene_writes;
			}
			// The proof belongs to one native recording sequence, never a merged
			// queue or secondary list's statistics.
			stats.direct_clear_depth_write = false;
			if (source_stats.best_copy_stats.vertices >= stats.best_copy_stats.vertices)
			{
				stats.best_copy_stats = source_stats.best_copy_stats;
				if (!demand.counters) stats.best_copy_stats.clear_counters();
			}
			if (source_stats.total.last_viewport.width * source_stats.total.last_viewport.height >=
				stats.total.last_viewport.width * stats.total.last_viewport.height)
				stats.total.last_viewport = source_stats.total.last_viewport;

			if (demand.clear_history)
				stats.clears.insert(stats.clears.end(), source_stats.clears.begin(), source_stats.clears.end());
			if (source_stats.copied_during_frame)
			{
				stats.copied_viewport = source_stats.copied_viewport;
				stats.copy_provenance_ambiguous = source_stats.copy_provenance_ambiguous;
				stats.capture_marker = source_stats.capture_marker;
				stats.depth_copy = source_stats.depth_copy;
				stats.capture_ticket = source_stats.capture_ticket;
			}

			stats.copied_during_frame |= source_stats.copied_during_frame;
			stats.reversed_clear_value = source_stats.reversed_clear_value;
			stats.clear_zero |= source_stats.clear_zero;
			stats.clear_one |= source_stats.clear_one;
			stats.clear_other |= source_stats.clear_other;
			stats.preservation_boundary |= source_stats.preservation_boundary;
		}
	}
};

struct capture_region
{
	uint32_t x = 0, y = 0, width = 0, height = 0;
	uint64_t layout_epoch = 0;
	bool operator==(const capture_region &other) const
	{
		return x == other.x && y == other.y && width == other.width && height == other.height && layout_epoch == other.layout_epoch;
	}
};

enum class depth_binding_state { waiting_game, waiting_activity, filtered, binding_failed, waiting_capture, ready };
enum class native_depth_ui_state { inactive, waiting_depth, calibrating, waiting_orientation, ready };

struct __declspec(uuid("c072221d-786d-4b6d-8f0b-52e0f325903e")) generic_depth_data
{
    // Runtime ownership, separate from retained backup/view lifetime. API
    // capture can suspend generic work without destroying in-flight resources.
    bool generic_capture_enabled = false;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
    uint64_t generic_capture_transitions = 0;
#endif
    uint64_t runtime_epoch = s_next_sample_scope.fetch_add(1, std::memory_order_relaxed);
    bool raw_scene_requested = false, raw_latest_available = false;
    uint64_t raw_basis_epoch = 0, raw_pending_capture_id = 0;
    sunshine_camera_binding::mailbox raw_sample_binding;
    sunshine_raw_scene::selected_frame raw_pending_metadata;
    sunshine_camera_binding::submission raw_latest_submission;
    sunshine_raw_scene::sample raw_latest_sample;
    uint64_t next_camera_observation_report = 0;
	struct raw_member_sample
	{
		uint64_t identity = 0, next_tick = 0;
		bool available = false;
		sunshine_raw_scene::sample sample;
		sunshine_camera_binding::submission submission;
	};
	std::array<raw_member_sample, 4> raw_members {};
	sunshine_depth::raw_source_roster raw_roster;
	sunshine_depth_capture::budget capture_budget;
	sunshine_depth_capture::capture_set captures;
	sunshine_depth_capture::capture_record selected_record;
	sunshine_depth::frame_depth selected_capture;
	resource anchor_depth_stencil {};
	uint64_t anchor_identity = 0;
	struct member_view
	{
		resource source {};
		uint64_t identity = 0, retire_after = 0;
		resource_view view {};
		uint64_t capture_after = 0;
	};
	// Four admitted views plus a bounded retirement window; exhausted slots wait.
	std::array<member_view, 8> member_views {};
	// The depth-stencil resource that is currently selected as being the main depth target
	resource selected_depth_stencil = { 0 };
	uint64_t selected_identity = 0;
	resource_desc selected_desc = {};
	bool capture_ready = false;
	bool native_access_open = false;
	bool native_driver = false;
	bool native_effects_lease = false;
	bool native_depth_requested = false;
	uint64_t native_access_present = 0;
	// Presentation-only snapshot: no borrowed GPU handles survive get_frame_depth.
	native_depth_ui_state native_ui_state = native_depth_ui_state::inactive;
	unsigned native_ui_samples = 0;
	uint64_t native_ui_present = 0;
	depth_binding_state binding_state = depth_binding_state::waiting_game;
	const char *binding_detail = "Waiting for the game to create a depth buffer.";
	const char *selection_reason = "waiting for depth";

	// Resource used to override automatic depth-stencil selection
	resource override_depth_stencil = { 0 };
	uint64_t override_identity = 0; // Resource lifetime; DLSS viewport changes retain the manual choice.
	sunshine_depth::manual_recovery_watch manual_recovery;
	bool manual_recovery_probing = false;

	// The current depth shader resource view bound to shaders
	// This can be created from either the selected depth-stencil resource (if it supports shader access) or from a backup resource
	resource_view selected_shader_resource = { 0 };

	// True when the shader resource view was created from the backup resource, false when it was created from the original depth-stencil
	bool using_backup_texture = false;

	// The view effects bind as DEPTH. ReShade 6.8 waits for the whole GPU queue
	// on every binding update, so it changes only when this view does, never
	// for a readiness flip. A logical source rotating through several physical
	// members (rotating) binds one stable copy per format and size instead of
	// each member's view. The add-on's renderer reads selected_shader_resource,
	// so that copy exists only while effects are enabled and an enabled technique
	// may read DEPTH (effects_read_depth); otherwise a rotating source binds nothing. Retired
	// copies are destroyed after the usual delay.
	resource_view effects_view = { 0 };
	bool effects_bound = false;
	bool rotating = false;
	bool stable_current = false; // The stable copy holds this Present's selected capture.
	// Any enabled technique (a safe over-approximation of a DEPTH reader).
	// Recounted at begin_effects after an effect reload or a technique toggle.
	bool effects_read_depth = false;
	bool effects_read_depth_dirty = true;
	struct stable_binding
	{
		resource texture = { 0 };
		resource_view view = { 0 };
		resource_desc desc = {};
		uint64_t destroy_after_frame = 0;
	};
	stable_binding stable;
	// Identity, sampled backup, frame and Present of the capture the stable copy
	// holds: an unchanged capture is not copied again.
	std::array<uint64_t, 4> stable_capture = {};
	std::vector<stable_binding> retired_stable;

	sunshine_depth::selection_policy selection;
	resource challenger_depth_stencil = { 0 };
	uint64_t challenger_id = 0;
	uint64_t challenger_started_frame = 0;
	uint64_t challenger_retired_until = 0;
	unsigned challenger_samples = 0;
	uint64_t probe_target_id = 0;
	uint64_t probe_started_tick = 0;
	sunshine_depth_probe::discovery_hint discovery_hint;
	resource sampled_depth_stencil = { 0 };
	uint64_t sampled_frame = 0;
	capture_region sampled_region;
	uint64_t next_sample_tick = 0;
	uint64_t last_current_sample_tick = 0;
	std::vector<uint64_t> observed_ids;
	std::unordered_map<uint64_t, capture_region> observed_regions;
	// Raw calibration summaries only; no textures or native pointers are cached.
	std::unordered_map<uint64_t, sunshine_depth::native_depth_calibration> native_calibrations;
	uint64_t last_logged_id = 0;
	bool last_logged_manual = false;
	depth_binding_state last_logged_state = depth_binding_state::waiting_game;
	sunshine_diagnostics::log_gate status_log;
	// Observation only: these fields never participate in routing or readiness.
	const char *capture_wait_detail = nullptr;
	struct capture_trace_record
	{
		uint64_t present = 0, frame = 0, anchor = 0, epoch = 0;
		unsigned count = 0, raster = 0, copied = 0, ambiguous = 0, confirmed = 0, views = 0, floors = 0;
		std::array<uint64_t, 4> members {};
		std::array<uint64_t, 4> last_used {}, draws {}, writes {};
		std::array<capture_region, 4> regions {};
		const char *reason = "not_observed";
	} trace_capture;
	uint64_t trace_effect_present = std::numeric_limits<uint64_t>::max();
	uint64_t trace_access_present = std::numeric_limits<uint64_t>::max();
	uint64_t trace_snapshot_frame = 0, trace_snapshot_present = 0;
	sunshine_diagnostics::family_gap_observer trace_gaps;
};

struct depth_stencil_backup
{
	// The number of effect runtimes referencing this backup
	uint32_t references = 1;

	// The depth-stencil that should be copied from
	resource depth_stencil = { 0 };

	// A resource used as target for a backup copy of this depth-stencil
	resource backup_texture = { 0 };
	// Assignment identity also rotates when pooled storage is bound to a new source.
	uint64_t content_identity = 0;
	// This allocation is a stable shader/probe presentation surface. D3D12
	// before-clear pixels are owned by capture_ticket, never by this cache.
	uint64_t published_ticket = 0, published_present = 0;
	sunshine_streamline::content::copy_snapshot published_depth_copy {};
	bool content_layout_supported = false;

	// Index of the frame after which the backup resource may be destroyed (used to delay destruction until resource is no longer in use)
	uint64_t destroy_after_frame = std::numeric_limits<uint64_t>::max();

	// Frame dimensions of the last effect runtime this backup was used with
	uint32_t frame_width = 0;
	uint32_t frame_height = 0;

	// Set to zero for automatic detection, otherwise will use the clear operation at the specific index within a frame
	uint32_t force_clear_index = 0;
	// Counted by recording threads under the shared lock (interlocked).
	LONG current_clear_index = 0;
};

SUNSHINE_DEPTH_ADDON_SHARED uint32_t configured_clear_index()
{
	uint32_t value = 0;
	reshade::get_config_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", value);
	return value;
}

struct depth_stencil_resource
{
	resource_desc desc;
	uint64_t identity = 0;
	bool api_identity = false;
	uint64_t layout_epoch = 1;
	sunshine_depth::depth_orientation_evidence orientation_evidence;
	capture_region preserved_region;
	bool have_preserved_region = false;

	depth_stencil_frame_stats last_frame_stats;

	// Index of the frame in which the depth-stencil was last/first seen used in
	uint64_t last_used_in_frame = std::numeric_limits<uint64_t>::max();
	uint64_t first_used_in_frame = std::numeric_limits<uint64_t>::max();
};

// Selection needs only these values. In particular, it does not consume the
// per-clear history, command provenance or orientation history retained by the
// recording/capture paths. Keep the unlocked snapshot contiguous and owned.
struct depth_selection_resource
{
	resource_desc desc;
	uint64_t identity, layout_epoch;
	capture_region preserved_region;
	bool have_preserved_region;
	struct frame_stats
	{
		draw_stats total;
		viewport copied_viewport;
		bool copied_during_frame, copy_provenance_ambiguous;
	} last_frame_stats;
	uint64_t last_used_in_frame, first_used_in_frame;

	explicit depth_selection_resource(const depth_stencil_resource &info) :
		desc(info.desc), identity(info.identity), layout_epoch(info.layout_epoch),
		preserved_region(info.preserved_region), have_preserved_region(info.have_preserved_region),
		last_frame_stats { info.last_frame_stats.total, info.last_frame_stats.copied_viewport,
			info.last_frame_stats.copied_during_frame, info.last_frame_stats.copy_provenance_ambiguous },
		last_used_in_frame(info.last_used_in_frame), first_used_in_frame(info.first_used_in_frame) {}
};
static_assert(std::is_trivially_copyable_v<depth_selection_resource>);
using depth_selection_snapshot = std::vector<std::pair<resource, depth_selection_resource>>;

SUNSHINE_DEPTH_ADDON_SHARED depth_selection_snapshot snapshot_depth_selection(
	const std::unordered_map<resource, depth_stencil_resource, resource_hash> &resources)
{
	depth_selection_snapshot result;
	result.reserve(resources.size());
	for (const auto &[resource, info] : resources)
		result.emplace_back(resource, depth_selection_resource(info));
	return result;
}

SUNSHINE_DEPTH_ADDON_SHARED auto find_depth_selection(const depth_selection_snapshot &snapshot, resource resource)
{
	return std::find_if(snapshot.begin(), snapshot.end(),
		[resource](const auto &entry) { return entry.first == resource; });
}

struct __declspec(uuid("94e283ef-6714-4d50-b936-71d6308a4be4")) generic_depth_device_data
{
    uint64_t native_identity{};
    const uint64_t device_epoch = s_next_sample_scope.fetch_add(1, std::memory_order_relaxed);
	// Draw/clear recording can run on other threads. A shared backup is still
	// needed if any effect runtime on this device uses generic/manual capture.
	std::atomic<unsigned> generic_capture_users { 0 };
	// UI requests expire at Present, including when the whole overlay is closed.
	// Bit 0 is the request since last Present, bit 1 is the current frame demand.
	std::atomic<unsigned> candidate_list_request { 0 };
	uint64_t frame_index = 0;
	uint64_t native_present_index = 0, native_depth_present_index = 0;
	// Callback ordinals, NOT game/frame-generation IDs. At most four short
	// diagnostic bursts; a scene-confirmed failure is required to start one.
	struct activity_trace_state
	{
		unsigned bursts = 0, effects = 0, accesses = 0, dropped_events = 0;
		uint64_t first_present = 0, last_present = 0, next_start_tick = 0;
		uint64_t event_present = 0, focus_source = 0;
	} activity_trace;

	// List of queues created for this device
	std::vector<command_queue *> queues;

	// List of all encountered depth-stencils of the last frame
	std::unordered_map<resource, depth_stencil_resource, resource_hash> depth_stencil_resources;

	// List of depth-stencils that should be tracked throughout each frame and potentially be backed up during clear operations
	std::vector<depth_stencil_backup> depth_stencil_backups;

	depth_stencil_backup *find_depth_stencil_backup(resource resource)
	{
		for (depth_stencil_backup &backup : depth_stencil_backups)
			if (backup.depth_stencil == resource)
				return &backup;
		return nullptr;
	}

	// Recording threads read a backup's frame size and clear index under the
	// shared lock (on_clear_depth_impl); the effects thread writes them here.
	static void configure_backup(depth_stencil_backup &backup, uint32_t width, uint32_t height,
		std::optional<uint32_t> force_clear_index = std::nullopt)
	{
		const std::unique_lock<std::shared_mutex> lock(s_mutex);
		backup.frame_width = width;
		backup.frame_height = height;
		if (force_clear_index) backup.force_clear_index = *force_clear_index;
	}

	depth_stencil_backup *track_depth_stencil_for_backup(device *device, resource depth_stencil, resource_desc desc)
	{
		assert(depth_stencil != 0);

		const auto it = std::find_if(depth_stencil_backups.begin(), depth_stencil_backups.end(),
			[depth_stencil](const depth_stencil_backup &existing) { return existing.depth_stencil == depth_stencil; });
		if (it != depth_stencil_backups.end())
		{
			depth_stencil_backup &backup = *it;
			backup.references++;

			return &backup;
		}

		const device_api api = device->get_api();
		if (api <= device_api::d3d12)
		{
			std::shared_lock<std::shared_mutex> lock(s_mutex);
			if (depth_stencil_resources.find(depth_stencil) == depth_stencil_resources.end())
				return nullptr;

			// Add reference to the resource so that it is not destroyed while it is being copied to the backup texture
			reinterpret_cast<IUnknown *>(depth_stencil.handle)->AddRef();
		}

		desc.type = resource_type::texture_2d;
		desc.heap = memory_heap::default_;
		desc.usage = resource_usage::shader_resource | resource_usage::copy_dest;

		if (desc.texture.samples > 1)
		{
			desc.texture.samples = 1;
			desc.usage |= resource_usage::resolve_dest;
		}

		if (api == device_api::d3d9)
			desc.texture.format = format::r32_float; // D3DFMT_R32F, since INTZ does not support D3DUSAGE_RENDERTARGET which is required for copying
		// Use depth format as-is in OpenGL and Vulkan, since those are valid for shader resource views there
		else if (api != device_api::opengl && api != device_api::vulkan)
			desc.texture.format = format_to_typeless(desc.texture.format);

		// First try to revive a backup resource that was previously enqueued for delayed destruction
		for (depth_stencil_backup &backup : depth_stencil_backups)
		{
			if (backup.depth_stencil != 0)
				continue;

			assert(backup.references == 0 && backup.destroy_after_frame != std::numeric_limits<uint64_t>::max() && backup.backup_texture != 0);

			const resource_desc existing_desc = device->get_resource_desc(backup.backup_texture);
			if (desc.texture.width == existing_desc.texture.width &&
				desc.texture.height == existing_desc.texture.height &&
				desc.texture.format == existing_desc.texture.format &&
				(desc.usage & existing_desc.usage) == desc.usage)
			{
				// Recording threads look backups up under the shared lock.
				const std::unique_lock<std::shared_mutex> lock(s_mutex);
				backup.references++;
				backup.depth_stencil = depth_stencil;
				backup.destroy_after_frame = std::numeric_limits<uint64_t>::max();
				backup.content_identity = sunshine_native_identity::allocate_identity();
				backup.published_ticket = backup.published_present = 0;
				backup.published_depth_copy = {};
				backup.content_layout_supported = content_supported(existing_desc);
				if (s_streamline_probe_events && api == device_api::d3d12)
					sunshine_streamline::depth_resource_initialized(backup.backup_texture.handle, backup.content_identity,
						device->get_native(), content_supported(existing_desc));

				return &backup;
			}
		}

		depth_stencil_backup new_backup;
		new_backup.depth_stencil = depth_stencil;

		if (device->create_resource(desc, nullptr, resource_usage::copy_dest, &new_backup.backup_texture))
		{
			device->set_resource_name(new_backup.backup_texture, "Sunshine depth presentation texture");
			new_backup.content_identity = sunshine_native_identity::allocate_identity();
			new_backup.content_layout_supported = content_supported(desc);
			if (s_streamline_probe_events && api == device_api::d3d12)
				sunshine_streamline::depth_resource_initialized(new_backup.backup_texture.handle, new_backup.content_identity,
					device->get_native(), content_supported(desc));

			// Growth can move every entry: exclude recording-thread lookups.
			const std::unique_lock<std::shared_mutex> lock(s_mutex);
			return &depth_stencil_backups.emplace_back(std::move(new_backup));
		}
		else
		{
			if (api <= device_api::d3d12)
				reinterpret_cast<IUnknown *>(depth_stencil.handle)->Release();
			sunshine_log::message(reshade::log::level::error, "Failed to create backup depth-stencil texture!");

			return nullptr;
		}
	}

	// Frames an unreferenced backup stays revivable before its delayed destruction.
	static constexpr uint64_t backup_retirement_frames = 50;

	void untrack_depth_stencil(device *device, resource depth_stencil, uint64_t retire_frames = backup_retirement_frames)
	{
		assert(depth_stencil != 0);

		const auto it = std::find_if(depth_stencil_backups.begin(), depth_stencil_backups.end(),
			[depth_stencil](const depth_stencil_backup &existing) { return existing.depth_stencil == depth_stencil; });
		if (it == depth_stencil_backups.end() || --it->references != 0)
			return;

		depth_stencil_backup &backup = *it;
		if (s_streamline_probe_events && device->get_api() == device_api::d3d12)
			sunshine_streamline::depth_resource_destroyed(backup.backup_texture.handle, backup.content_identity);
		{
			// Recording threads look backups up by this source under the shared lock.
			const std::unique_lock<std::shared_mutex> lock(s_mutex);
			backup.depth_stencil = { 0 };
		}

		// Do not destroy backup texture immediately since it may still be referenced by a command list that is in flight or was prerecorded
		// Instead mark it for delayed destruction in the future
		backup.destroy_after_frame = frame_index + retire_frames;

		const device_api api = device->get_api();
		if (api <= device_api::d3d12)
		{
			// Release the reference that was added above
			reinterpret_cast<IUnknown *>(depth_stencil.handle)->Release();
		}
	}
};

#ifdef __MINGW32__
__CRT_UUID_DECL(state_tracking, 0x806c3b54, 0xbaf6, 0x46a7, 0xa7, 0x2a, 0x28, 0xc6, 0x29, 0x14, 0x3d, 0x84)
__CRT_UUID_DECL(generic_depth_data, 0xc072221d, 0x786d, 0x4b6d, 0x8f, 0x0b, 0x52, 0xe0, 0xf3, 0x25, 0x90, 0x3e)
__CRT_UUID_DECL(generic_depth_device_data, 0x94e283ef, 0x6714, 0x4d50, 0xb9, 0x36, 0x71, 0xd6, 0x30, 0x8a, 0x4b, 0xe4)
#endif

SUNSHINE_DEPTH_ADDON_SHARED void request_depth_candidate_list(generic_depth_device_data *data)
{
	if (data) data->candidate_list_request.fetch_or(3, std::memory_order_relaxed);
}

SUNSHINE_DEPTH_ADDON_SHARED void consume_depth_candidate_list_request(generic_depth_device_data &data)
{
	auto request = data.candidate_list_request.load(std::memory_order_relaxed);
	while (!data.candidate_list_request.compare_exchange_weak(request, (request & 1) ? 2u : 0u,
		std::memory_order_relaxed)) {}
}

SUNSHINE_DEPTH_ADDON_SHARED depth_tracking_demand tracking_demand(const generic_depth_device_data &data)
{
	const bool visible = data.candidate_list_request.load(std::memory_order_relaxed) != 0;
	return {visible || s_frame_activity_trace || data.generic_capture_users.load(std::memory_order_relaxed) != 0, visible};
}

SUNSHINE_DEPTH_ADDON_SHARED depth_tracking_demand tracking_demand(device *device)
{
	const auto *data = device->get_private_data<generic_depth_device_data>();
	return data ? tracking_demand(*data) : depth_tracking_demand {};
}

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
// Defined in depth_addon.cpp.
template <typename DepthInfo>
capture_region effective_capture_region(const DepthInfo &info);
template <typename DepthInfo>
void observe_candidate(generic_depth_data &data, const DepthInfo &info, uint32_t width, uint32_t height);
void prune_depth_selection(generic_depth_data &data, const std::vector<uint64_t> &eligible_ids);
void observe_preserved_layout(depth_stencil_resource &info, const depth_stencil_frame_stats &stats);
template <typename DepthInfo>
bool candidate_matches_shape(const DepthInfo &info, uint32_t width, uint32_t height);
bool validate_capture_record(effect_runtime *runtime, const generic_depth_data &data,
	const generic_depth_device_data &device_data, const sunshine_depth_capture::capture_record &record);
void on_clear_depth_impl(command_list *cmd_list, state_tracking &state, resource depth_stencil, clear_op op);
bool on_draw(command_list *cmd_list, uint32_t vertices, uint32_t instances, uint32_t, uint32_t);
bool on_draw_indexed(command_list *cmd_list, uint32_t indices, uint32_t instances, uint32_t, int32_t, uint32_t);
bool on_draw_indirect(command_list *cmd_list, indirect_command type, resource, uint64_t, uint32_t draw_count, uint32_t);
bool on_dispatch_mesh(command_list *cmd_list, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
bool on_copy_resource(command_list *cmd_list, resource, resource dest);
bool on_copy_texture_region(command_list *cmd_list, resource, uint32_t, const subresource_box *source_box,
	resource dest, uint32_t subresource, const subresource_box *dest_box, filter_mode);
bool on_copy_buffer_to_texture(command_list *cmd_list, resource, uint64_t, uint32_t, uint32_t,
	resource dest, uint32_t subresource, const subresource_box *dest_box);
bool on_resolve_texture_region(command_list *cmd_list, resource, uint32_t, const subresource_box *source_box,
	resource dest, uint32_t subresource, uint32_t, uint32_t, uint32_t, format);
void set_manual_depth_override(generic_depth_data &data, resource depth_stencil, uint64_t identity);
sunshine_streamline::provider::source_status describe_streamline_source(effect_runtime *runtime);
#endif

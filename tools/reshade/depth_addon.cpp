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

#include "addon_lifetime.h"
#include <Windows.h>
#include <TlHelp32.h>
#include <Unknwn.h>
#include <d3d12.h>
#include <imgui.h>
#include <reshade.hpp>
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
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
#include <chrono>
#include <stdexcept>
#include "depth_selection_snapshot_test.h"
#endif
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

using namespace reshade::api;

static std::shared_mutex s_mutex;
static bool s_sunshine_auto = true;
static bool s_registered = false;
static bool s_initialized = false;
// Fixed for each add-on registration; changing the diagnostic requires restart.
static bool s_streamline_probe_events = false;
static bool s_streamline_source_events = false;
static bool s_frame_activity_trace = false; // Opt-in metadata only; never changes capture policy.
static const char *s_status_message = "Depth selection has not been initialized.";
static constexpr const char *s_overlay_title = "Sunshine 3D";
static std::atomic<uint64_t> s_next_command_identity { 1 };
static std::atomic<uint64_t> s_next_sample_scope { 1 };

static bool content_supported(const resource_desc &desc)
{
	return desc.type == resource_type::texture_2d && desc.texture.levels == 1 &&
		desc.texture.depth_or_layers == 1 && desc.texture.samples == 1;
}

// One challenger may be tracked across all runtimes/devices. The sampler also
// permits only one globally in-flight GPU readback. Ownership is short-lived
// and no native resource address is ever saved in configuration.
static effect_runtime *s_challenger_owner = nullptr;
static std::mutex s_challenger_mutex;

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

static bool s_disable_intz = false;
// Enable or disable the creation of backup copies at clear operations on the selected depth-stencil
static unsigned int s_preserve_depth_buffers = 0;
// Choose impact of draw statistics in the detection heuristic
static draw_stats_heuristic s_draw_stats_heuristic = draw_stats_heuristic::prefer_vertices;
// Enable or disable the aspect ratio check from 'check_aspect_ratio' in the detection heuristic
static aspect_ratio_heuristic s_aspect_ratio_heuristic = aspect_ratio_heuristic::similar_aspect_ratio;
// Enable or disable the format check from 'check_depth_format' in the detection heuristic
static unsigned int s_format_filtering = 0;
static unsigned int s_custom_resolution_filtering[2] = {};

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
    struct provided_member {
      resource source{};
      uint64_t identity{}, capture_after{}, last_present{};
    };
    // Source authority is API-owned; these entries only keep the shared
    // preservation assignments alive across physical buffer rotation.
    std::array<provided_member, 8> provided_members{};
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
	uint32_t current_clear_index = 0;
};

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

static depth_selection_snapshot snapshot_depth_selection(
	const std::unordered_map<resource, depth_stencil_resource, resource_hash> &resources)
{
	depth_selection_snapshot result;
	result.reserve(resources.size());
	for (const auto &[resource, info] : resources)
		result.emplace_back(resource, depth_selection_resource(info));
	return result;
}

static auto find_depth_selection(const depth_selection_snapshot &snapshot, resource resource)
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

			return &depth_stencil_backups.emplace_back(std::move(new_backup));
		}
		else
		{
			if (api <= device_api::d3d12)
				reinterpret_cast<IUnknown *>(depth_stencil.handle)->Release();
			reshade::log::message(reshade::log::level::error, "Failed to create backup depth-stencil texture!");

			return nullptr;
		}
	}

	void untrack_depth_stencil(device *device, resource depth_stencil)
	{
		assert(depth_stencil != 0);

		const auto it = std::find_if(depth_stencil_backups.begin(), depth_stencil_backups.end(),
			[depth_stencil](const depth_stencil_backup &existing) { return existing.depth_stencil == depth_stencil; });
		if (it == depth_stencil_backups.end() || --it->references != 0)
			return;

		depth_stencil_backup &backup = *it;
		if (s_streamline_probe_events && device->get_api() == device_api::d3d12)
			sunshine_streamline::depth_resource_destroyed(backup.backup_texture.handle, backup.content_identity);
		backup.depth_stencil = { 0 };

		// Do not destroy backup texture immediately since it may still be referenced by a command list that is in flight or was prerecorded
		// Instead mark it for delayed destruction in the future
		backup.destroy_after_frame = frame_index + 50; // Destroy after 50 frames

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

static void request_depth_candidate_list(generic_depth_device_data *data)
{
	if (data) data->candidate_list_request.fetch_or(3, std::memory_order_relaxed);
}

static void consume_depth_candidate_list_request(generic_depth_device_data &data)
{
	auto request = data.candidate_list_request.load(std::memory_order_relaxed);
	while (!data.candidate_list_request.compare_exchange_weak(request, (request & 1) ? 2u : 0u,
		std::memory_order_relaxed)) {}
}

static depth_tracking_demand tracking_demand(const generic_depth_device_data &data)
{
	const bool visible = data.candidate_list_request.load(std::memory_order_relaxed) != 0;
	return {visible || s_frame_activity_trace || data.generic_capture_users.load(std::memory_order_relaxed) != 0, visible};
}

static depth_tracking_demand tracking_demand(device *device)
{
	const auto *data = device->get_private_data<generic_depth_device_data>();
	return data ? tracking_demand(*data) : depth_tracking_demand {};
}

// The resource inventory remains owned by ReShade's device lifetime. API
// adapters only ask whether that inventory can preserve an exact native object.
static std::unordered_map<uint64_t, generic_depth_device_data *> s_preservation_devices;

static auto find_preserved_source(generic_depth_device_data &device_data, uint64_t native, uint64_t identity)
{
	auto found = device_data.depth_stencil_resources.find(resource{native});
	if (found != device_data.depth_stencil_resources.end() && found->second.identity == identity) return found;
	// A proven shared cookie also handles an interface wrapper whose address
	// differs. Dimensions and contents are never used as identity evidence.
	return std::find_if(device_data.depth_stencil_resources.begin(), device_data.depth_stencil_resources.end(),
		[identity](const auto &entry) { return identity && entry.second.identity == identity; });
}

static bool preservation_available(uint64_t native, uint64_t identity, uint64_t device_identity)
{
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	const auto device = s_preservation_devices.find(device_identity);
	if (device == s_preservation_devices.end() || !identity) return false;
	const auto found = find_preserved_source(*device->second, native, identity);
	if (found == device->second->depth_stencil_resources.end() || !found->second.api_identity || !content_supported(found->second.desc)) return false;
	if (!found->second.last_frame_stats.total.has_work()) return false;
	// Mode 2 cannot take an end-of-frame copy. A registered DSV that is only
	// populated by clears/copies may have no legal preservation boundary at all;
	// it keeps the native API snapshot route, without changing source authority.
	return s_preserve_depth_buffers != 2 || found->second.last_frame_stats.preservation_boundary;
}

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
		reshade::log::message(reshade::log::level::info, out.text);
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
	reshade::log::message(reshade::log::level::info, out.text);
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

static void clear_raw_sample_request(generic_depth_data &data)
{
	data.raw_scene_requested = false;
	data.raw_basis_epoch = 0;
	data.raw_pending_capture_id = 0;
	data.raw_sample_binding.cancel();
	data.raw_pending_metadata = {};
	data.raw_latest_available = false;
	data.raw_latest_submission = {};
	data.raw_latest_sample = {};
	data.raw_members = {};
}

static uint64_t depth_lifetime(device *device, resource value)
{
	const auto *data = device->get_private_data<generic_depth_device_data>();
	if (data == nullptr || value == 0) return 0;
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	const auto found = data->depth_stencil_resources.find(value);
	if (s_streamline_probe_events && found != data->depth_stencil_resources.end() && device->get_api() == device_api::d3d12)
		sunshine_streamline::depth_resource_initialized(value.handle, found->second.identity, device->get_native(),
			content_supported(found->second.desc));
	return found != data->depth_stencil_resources.end() ? found->second.identity : 0;
}

static bool check_depth_format(format format)
{
	switch (s_format_filtering)
	{
	case 1:
		return format == format::d16_unorm || format == format::r16_typeless;
	case 2:
		return format == format::d16_unorm_s8_uint;
	case 3:
		return format == format::d24_unorm_x8_uint;
	case 4:
		return format == format::d24_unorm_s8_uint || format == format::r24_g8_typeless;
	case 5:
		return format == format::d32_float || format == format::r32_float || format == format::r32_typeless;
	case 6:
		return format == format::d32_float_s8_uint || format == format::r32_g8_typeless;
	case 7:
		return format == format::intz;
	default:
		return false;
	}
}
// Checks whether the aspect ratio of the two sets of dimensions is similar or not
static bool check_aspect_ratio(float width_to_check, float height_to_check, float width, float height)
{
	if (width_to_check == 0.0f || height_to_check == 0.0f)
		return true;

	if (s_aspect_ratio_heuristic == aspect_ratio_heuristic::match_resolution_exactly || (s_aspect_ratio_heuristic == aspect_ratio_heuristic::match_custom_resolution_exactly && s_custom_resolution_filtering[0] == 0 && s_custom_resolution_filtering[1] == 0))
		return width_to_check == width && height_to_check == height;
	if (s_aspect_ratio_heuristic == aspect_ratio_heuristic::match_custom_resolution_exactly)
		return width_to_check == s_custom_resolution_filtering[0] && height_to_check == s_custom_resolution_filtering[1];

	float w_ratio = width / width_to_check;
	float h_ratio = height / height_to_check;
	const float aspect_ratio_delta = (width / height) - (width_to_check / height_to_check);

	// Upscalers render scene depth below output resolution; DLSS Performance and
	// Ultra Performance are outside the original 1.85 output/input ratio limit.
	// Only extend the default mode. Explicit exact/custom/multiple filters retain
	// their established meaning, and the existing larger-buffer cases below stay.
	if (s_sunshine_auto && s_aspect_ratio_heuristic == aspect_ratio_heuristic::similar_aspect_ratio &&
		sunshine_depth::auto_candidate_shape(static_cast<uint32_t>(width_to_check), static_cast<uint32_t>(height_to_check),
			static_cast<uint32_t>(width), static_cast<uint32_t>(height)))
		return true;

	// Accept if dimensions are similar in value or almost exact multiples
	return std::abs(aspect_ratio_delta) <= 0.1f && ((w_ratio <= 1.85f && w_ratio >= 0.5f && h_ratio <= 1.85f && h_ratio >= 0.5f) ||
		(s_aspect_ratio_heuristic == aspect_ratio_heuristic::multiples_of_resolution && std::modf(w_ratio, &w_ratio) <= 0.02f && std::modf(h_ratio, &h_ratio) <= 0.02f));
}

static bool valid_depth_viewport(const viewport &view, const resource_desc &desc)
{
	return std::isfinite(view.x) && std::isfinite(view.y) && std::isfinite(view.width) && std::isfinite(view.height) &&
		view.x >= 0 && view.y >= 0 && view.width >= 1 && view.height >= 1 &&
		view.x + view.width <= desc.texture.width + 0.5f && view.y + view.height <= desc.texture.height + 0.5f;
}

template <typename DepthInfo>
static capture_region effective_capture_region(const DepthInfo &info)
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
static void observe_candidate(generic_depth_data &data, const DepthInfo &info, uint32_t width, uint32_t height)
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

static void prune_depth_selection(generic_depth_data &data, const std::vector<uint64_t> &eligible_ids)
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

static void observe_preserved_layout(depth_stencil_resource &info, const depth_stencil_frame_stats &stats)
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
static bool candidate_matches_shape(const DepthInfo &info, uint32_t width, uint32_t height)
{
	if (s_aspect_ratio_heuristic == aspect_ratio_heuristic::none ||
		check_aspect_ratio(static_cast<float>(info.desc.texture.width), static_cast<float>(info.desc.texture.height),
			static_cast<float>(width), static_cast<float>(height)))
		return true;

	// Padded allocations may contain a smaller active DLSS/FSR viewport. Keep
	// explicit filters exact and admit this evidence only in the default mode.
	const auto region = effective_capture_region(info);
	return s_sunshine_auto && s_aspect_ratio_heuristic == aspect_ratio_heuristic::similar_aspect_ratio &&
		check_aspect_ratio(static_cast<float>(region.width), static_cast<float>(region.height), static_cast<float>(width), static_cast<float>(height));
}

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
	auto displayed_copy = info.last_frame_stats.depth_copy;
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
			if (presentation) displayed_copy = backup->published_depth_copy;
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
	auto &depth = result.metadata;
	depth.source_resource = source; depth.resource = sampled; depth.shader_resource = view;
	depth.width = out.shape.width; depth.height = out.shape.height;
	depth.x = region.x; depth.y = region.y; depth.active_width = region.width; depth.active_height = region.height;
	depth.source_id = identity; depth.backup_id = using_backup ? assignment : 0;
	depth.layout_epoch = info.layout_epoch; depth.frame_index = info.last_used_in_frame; depth.runtime_epoch = data.runtime_epoch;
	depth.detected_orientation = info.orientation_evidence.detected();
	depth.orientation_agreeing_frames = info.orientation_evidence.agreeing_frames();
	depth.capture_marker = displayed_copy.recording; depth.depth_copy = displayed_copy;
	depth.ready = out.current(device_data.native_present_index, device_data.frame_index, data.runtime_epoch);
	lock.unlock();
	if (view != 0 && runtime->get_device()->get_resource_from_view(view) != sampled) return {};
	return result;
}

static bool validate_capture_record(effect_runtime *runtime, const generic_depth_data &data,
	const generic_depth_device_data &device_data, const sunshine_depth_capture::capture_record &record)
{
	const auto current = capture_record(runtime, data, device_data, resource{record.source}, record.identity,
		resource_view{record.view}, !record.direct, record.capture_after).record;
	return current.current(device_data.native_present_index, device_data.frame_index, data.runtime_epoch) &&
		current.sampled == record.sampled && current.assignment == record.assignment && current.layout == record.layout &&
		current.shape == record.shape && current.frame == record.frame && current.present == record.present;
}

// A stable shader/probe surface receives an admitted snapshot. It never owns
// before-clear pixels: those resources and read leases belong to capture owner.
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
	sunshine_streamline::content::copy_snapshot presented;
	if (s_streamline_probe_events) {
		sunshine_streamline::depth_resource_initialized(destination.handle, assignment, runtime->get_device()->get_native(), true);
		presented = sunshine_streamline::forward_depth_copy(commands->get_native(), current.metadata.depth_copy, {destination.handle, assignment});
	}
	const std::unique_lock<std::shared_mutex> lock(s_mutex);
	const auto found = device_data.depth_stencil_resources.find(source);
	auto *view = device_data.find_depth_stencil_backup(source);
	if (found == device_data.depth_stencil_resources.end() || found->second.identity != identity || !view ||
		view->content_identity != assignment || found->second.last_frame_stats.capture_ticket.id != ticket.id ||
		device_data.frame_index != frame || device_data.native_present_index != present) return;
	view->published_ticket = ticket.id;
	view->published_present = present;
	view->published_depth_copy = presented;
}

static void preserve_d3d12_at_effects(effect_runtime *runtime, command_list *commands,
	generic_depth_device_data &device_data, resource source, uint64_t identity, uint64_t capture_after, bool publish)
{
	const auto frame = device_data.frame_index, present = device_data.native_present_index;
	viewport copied_viewport{};
	resource_usage before{};
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
		capture = !info.last_frame_stats.copied_during_frame && !info.last_frame_stats.preservation_boundary &&
			!info.last_frame_stats.preservation_reservation && s_preserve_depth_buffers != 2 && (info.desc.usage & resource_usage::copy_source) != 0;
		if (capture) {
			static std::atomic<uint64_t> next_reservation{1};
			reservation = next_reservation.fetch_add(1, std::memory_order_relaxed);
			info.last_frame_stats.preservation_reservation = reservation;
			info.last_frame_stats.copy_provenance_ambiguous = true;
		}
		before = info.desc.usage & (resource_usage::depth_stencil | resource_usage::shader_resource);
		copied_viewport = info.last_frame_stats.total.last_viewport;
	}
	if (capture)
	{
		// A missed before-clear boundary is never recoverable at end of frame.
		uint32_t native_before = 0;
		if ((before & resource_usage::depth_stencil_write) != 0) native_before = D3D12_RESOURCE_STATE_DEPTH_WRITE;
		else
		{
			if ((before & resource_usage::depth_stencil_read) != 0) native_before |= D3D12_RESOURCE_STATE_DEPTH_READ;
			if ((before & resource_usage::shader_resource_pixel) != 0) native_before |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
			if ((before & resource_usage::shader_resource_non_pixel) != 0) native_before |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
		}
		auto ticket = sunshine_streamline::depth_capture::record_preserved(commands->get_native(), source.handle, native_before);
		sunshine_streamline::content::copy_snapshot copied;
		if (ticket && s_streamline_probe_events) {
			sunshine_streamline::depth_resource_initialized(ticket.texture, ticket.resource_id, runtime->get_device()->get_native(), true);
			copied = sunshine_streamline::record_depth_copy(commands->get_native(), {source.handle, identity}, {ticket.texture, ticket.resource_id});
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
			stats.depth_copy = copied; stats.capture_marker = copied.recording;
			stats.copied_during_frame = true; stats.copied_viewport = copied_viewport;
			observe_preserved_layout(found->second, stats);
		}
	}
	if (publish) publish_preserved_depth(runtime, commands, device_data, source, identity, capture_after);
}

// Produce an allowed end-of-frame copy before consumers decide which current
// source can render. Mode 2's explicit-API aliasing restriction still applies.
// Probe, capture-member and legacy selected paths share the same live facts;
// a selection snapshot cannot authorize a second copy or an inactive present.
static void preserve_depth_at_effects(effect_runtime *runtime, command_list *cmd_list,
	generic_depth_device_data &device_data, resource source, uint64_t identity, uint64_t capture_after = 0, bool publish = true)
{
	device *const device = runtime->get_device();
	const auto api = device->get_api();
	if (api == device_api::d3d12)
	{
		preserve_d3d12_at_effects(runtime, cmd_list, device_data, source, identity, capture_after, publish);
		return;
	}
	if (s_preserve_depth_buffers == 2 && (api == device_api::d3d12 || api == device_api::vulkan)) return;
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
			info.last_frame_stats.preservation_boundary ||
			(info.desc.usage & (resource_usage::copy_source | resource_usage::resolve_source)) == 0) return;
		// Once a post-scene preservation boundary was missed, the original may
		// already be cleared or reused. End-of-frame copying cannot recover those
		// pixels. Wait for a current preserved copy instead of sampling a clear.
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
	observe_preserved_layout(found->second, stats);
	stats.depth_copy = {};
	if (s_streamline_probe_events && api == device_api::d3d12)
	{
		sunshine_streamline::depth_resource_initialized(source.handle, identity, device->get_native(), content_supported(desc));
		sunshine_streamline::depth_resource_initialized(backup_texture.handle, assignment, device->get_native(), backup->content_layout_supported);
		stats.depth_copy = sunshine_streamline::record_depth_copy(cmd_list->get_native(), { source.handle, identity }, { backup_texture.handle, assignment });
	}
	stats.capture_marker = stats.depth_copy.recording;
}

static void retire_provided_members(effect_runtime *runtime, generic_depth_data &data,
	generic_depth_device_data &device_data, bool all = false)
{
	for (auto &member : data.provided_members)
		if (member.identity && (all || device_data.native_present_index > member.last_present + 50))
		{
			device_data.untrack_depth_stencil(runtime->get_device(), member.source);
			member = {};
		}
}

static void set_generic_capture_enabled(effect_runtime *runtime, generic_depth_data &data, bool enabled);

bool sunshine_depth::copy_selected_depth(effect_runtime *runtime, command_list *commands,
	const sunshine_streamline::depth_capture::packet &selection, resource destination, frame_depth &captured,
	sunshine_streamline::depth_capture::consumer_diagnostic *diagnostic)
{
	captured = {};
	constexpr uint32_t sampled_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	if (!selection.shared_preservation)
		return runtime && commands && destination.handle && sunshine_streamline::depth_capture::copy_current(
			commands->get_native(), selection, destination.handle, sampled_state, diagnostic);
	const auto &source = selection.metadata;
	const auto identity = selection.resource_id;
	if (!runtime || !commands || !identity || !destination.handle) return false;
	auto *device = runtime->get_device();
	auto *device_data = device->get_private_data<generic_depth_device_data>();
	auto *data = runtime->get_private_data<generic_depth_data>();
	if (!device_data || !data || device->get_api() != device_api::d3d12) return false;
	// A newly API-selected tracked source needs preservation before this copy.
	// Native snapshots never enter here and leave Generic capture dormant.
	set_generic_capture_enabled(runtime, *data, true);
	resource original{};
	resource_desc desc;
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		const auto found = find_preserved_source(*device_data, source.resource.native, identity);
		if (found == device_data->depth_stencil_resources.end() || !found->second.api_identity ||
			!content_supported(found->second.desc)) return false;
		original = found->first; desc = found->second.desc;
	}
	if (desc.texture.width != source.resource.width || desc.texture.height != source.resource.height) return false;
	auto member = std::find_if(data->provided_members.begin(), data->provided_members.end(),
		[identity](const auto &entry) { return entry.identity == identity; });
	if (member == data->provided_members.end())
	{
		member = std::find_if(data->provided_members.begin(), data->provided_members.end(),
			[](const auto &entry) { return !entry.identity; });
		if (member == data->provided_members.end()) return false;
		const bool existing = device_data->find_depth_stencil_backup(original) != nullptr;
		auto *backup = device_data->track_depth_stencil_for_backup(device, original, desc);
		if (!backup) return false;
		backup->frame_width = source.resource.area.width;
		backup->frame_height = source.resource.area.height;
		*member = {original, identity, existing ? 0 : device_data->frame_index, device_data->native_present_index};
	}
	member->last_present = device_data->native_present_index;
	// Shared preservation owns all timing/readiness decisions, including the
	// mode-2 requirement to copy before clear and the fresh-assignment boundary.
	preserve_depth_at_effects(runtime, commands, *device_data, original, identity, member->capture_after, false);
	const auto current = capture_record(runtime, *data, *device_data, original, identity, {}, true, member->capture_after, false);
	if (!current.record.current(device_data->native_present_index, device_data->frame_index, data->runtime_epoch)) return false;
	sunshine_streamline::depth_capture::preservation_ticket ticket;
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		const auto found = find_preserved_source(*device_data, original.handle, identity);
		if (found == device_data->depth_stencil_resources.end()) return false;
		ticket = found->second.last_frame_stats.capture_ticket;
		if (!ticket || ticket.texture != current.record.sampled || ticket.resource_id != current.record.assignment) return false;
	}
	if (!sunshine_streamline::depth_capture::copy_preserved(commands->get_native(), runtime->get_command_queue()->get_native(),
		ticket, destination.handle, sampled_state, diagnostic)) return false;
	const auto after = capture_record(runtime, *data, *device_data, original, identity, {}, true, member->capture_after, false);
	if (!after.record.current(device_data->native_present_index, device_data->frame_index, data->runtime_epoch) ||
		after.record.present != current.record.present || after.record.frame != current.record.frame ||
		after.record.sampled != current.record.sampled || after.record.assignment != current.record.assignment ||
		after.record.layout != current.record.layout || !(after.record.shape == current.record.shape)) return false;
	captured = current.metadata;
	return true;
}

static generic_depth_data::member_view *select_captured_depth(effect_runtime *runtime, command_list *cmd_list, generic_depth_data &data,
	generic_depth_device_data &device_data, const depth_selection_snapshot &snapshot, bool automatic, bool manual,
	uint32_t frame_width, uint32_t frame_height, resource &selected, const depth_selection_resource *&selected_info)
{
	data.capture_wait_detail = nullptr;
	retire_member_views(runtime, data, device_data);
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
	const uint64_t anchor = selected_info ? selected_info->identity : 0;
	const auto group = data.capture_budget.update(anchor, observations, manual || !automatic);
	data.anchor_depth_stencil = selected;
	data.anchor_identity = anchor;
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
			backup->frame_width = frame_width; backup->frame_height = frame_height;
			if (s_preserve_depth_buffers) reshade::get_config_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", backup->force_clear_index);
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
	// copy. In mode 1 a draw without a trailing clear may have no earlier copy.
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
	preferred.anchor = anchor; preferred.manual = manual || !automatic;
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
	if (automatic && !manual && data.raw_scene_requested &&
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
	if (s_frame_activity_trace)
	{
		auto &trace = data.trace_capture;
		trace = {};
		trace.present = device_data.native_present_index; trace.frame = device_data.frame_index;
		trace.anchor = anchor; trace.epoch = group.epoch; trace.count = group.count; trace.members = group.members;
		trace.reason = sunshine_depth_capture::name(decision.reason);
		for (unsigned i = 0; i != group.count; ++i)
		{
			const auto found = std::find_if(observations.begin(), observations.end(), [&](const auto &o) { return o.id == group.members[i]; });
			if (found != observations.end())
			{
				if (found->qualified) trace.confirmed |= 1u << i;
			}
			const auto source = std::find_if(snapshot.begin(), snapshot.end(), [&](const auto &entry) { return entry.second.identity == group.members[i]; });
			if (source != snapshot.end())
			{
				trace.last_used[i] = source->second.last_used_in_frame;
				trace.draws[i] = source->second.last_frame_stats.total.drawcalls;
				trace.writes[i] = source->second.last_frame_stats.total.scene_writes;
				if (source->second.last_used_in_frame == device_data.frame_index && source->second.last_frame_stats.total.drawcalls) trace.raster |= 1u << i;
			}
			for (unsigned j = 0; j != data.captures.count; ++j)
			{
				const auto &record = data.captures.records[j];
				if (record.identity != group.members[i]) continue;
				trace.regions[i] = {record.shape.x, record.shape.y, record.shape.extent_width, record.shape.extent_height, record.layout};
				if (record.frame == device_data.frame_index && record.preserved) trace.copied |= 1u << i;
				if (record.ambiguous) trace.ambiguous |= 1u << i;
			}
			for (const auto &entry : data.member_views)
				if (entry.identity == group.members[i] && entry.view != 0)
				{
					trace.views |= 1u << i;
					if (entry.capture_after && device_data.frame_index <= entry.capture_after) trace.floors |= 1u << i;
				}
		}
	}
	if (data.raw_scene_requested || (!manual && automatic && group.count > 1))
	{
		const depth_selection_resource *current_info = nullptr;
		resource current_source {};
		const auto current_id = decision.source;
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
				return nullptr; // Repeated effects/manual changes cannot bypass the new-assignment floor.
			}
			return &entry;
		}
	return nullptr;
}

static void release_sample_reference(effect_runtime *runtime, generic_depth_data &data)
{
	if (data.sampled_depth_stencil != 0)
	{
		if (auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>())
			device_data->untrack_depth_stencil(runtime->get_device(), data.sampled_depth_stencil);
		data.sampled_depth_stencil = { 0 };
	}
}

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
			device_data->untrack_depth_stencil(runtime->get_device(), data.challenger_depth_stencil);
			// The inherited capture path may have recorded work on other queues.
			// Retain its usual retirement interval before allocating another extra
			// backup, unless a nonretiring capture slot took over this allocation.
			// An in-flight sampler reference alone cannot skip that interval.
			if (!retained) data.challenger_retired_until = device_data->frame_index + 50;
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

static void set_generic_capture_enabled(effect_runtime *runtime, generic_depth_data &data, bool enabled)
{
	if (data.generic_capture_enabled == enabled)
		return;
	auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (device_data == nullptr)
		return;
	data.generic_capture_enabled = enabled;
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
	++data.generic_capture_transitions;
#endif
	if (enabled)
		device_data->generic_capture_users.fetch_add(1, std::memory_order_relaxed);
	else
	{
		[[maybe_unused]] const auto previous = device_data->generic_capture_users.fetch_sub(1, std::memory_order_relaxed);
		assert(previous != 0);
		// This uses the existing delayed backup retirement. An already submitted
		// sampler retains its separate reference until normal completion polling.
		release_challenger(runtime, data, true);
		data.capture_ready = false;
		data.captures = {};
		data.selected_record = {};
		data.selected_capture = {};
		// Keep bounded selected/member allocations and CPU preference history for
		// manual pins/fallback. Their GPU copies are gated below; no wait/reset or
		// unsafe immediate view/resource destruction is required for API handover.
	}
}

static const char *content_kind_name(sunshine_depth::content_kind kind)
{
	switch (kind)
	{
	case sunshine_depth::content_kind::useful: return "scene";
	case sunshine_depth::content_kind::flat: return "flat";
	case sunshine_depth::content_kind::unreliable: return "unreliable";
	default: return "unmeasured";
	}
}

static const char *binding_state_name(depth_binding_state state)
{
	switch (state)
	{
	case depth_binding_state::ready: return "ready";
	case depth_binding_state::waiting_activity: return "waiting for rendering";
	case depth_binding_state::filtered: return "no eligible buffer";
	case depth_binding_state::binding_failed: return "binding failed";
	case depth_binding_state::waiting_capture: return "waiting for captured depth";
	default: return "waiting for depth buffer";
	}
}

static const char *const format_to_string(format value);

static void on_clear_depth_impl(command_list *cmd_list, state_tracking &state, resource depth_stencil, clear_op op)
{
	if (depth_stencil == 0)
		return;

	device *const device = cmd_list->get_device();
	auto *device_data = device->get_private_data<generic_depth_device_data>();
	if (device_data == nullptr) return;
	if (op == clear_op::direct_depth_switch &&
		(device->get_api() != device_api::d3d12 || !content_supported(device->get_resource_desc(depth_stencil))))
		return; // One depth clear only proves the one supported depth subresource.

	// If this is queue state (happens if this is a immediate command list), need to protect access to it, since another thread may be in a present call, which can reset it
	std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
	if (state.is_queue)
		lock.lock();

	depth_stencil_frame_stats &stats = state.stats_per_used_depth_stencil[depth_stencil];
	const auto demand = tracking_demand(*device_data);
	stats.limit_tracking(demand);

	// Ignore clears when there was no meaningful workload (e.g. at the start of a frame)
	// Don't do this in Vulkan, to handle common case of DXVK flushing its immediate command buffer and thus resetting its stats during the frame
	if (!stats.current.has_work() && (device->get_api() != device_api::vulkan))
		return;
	stats.preservation_boundary = true;
	if (device_data->generic_capture_users.load(std::memory_order_relaxed) == 0) return;
	depth_stencil_backup *const depth_stencil_backup = device_data->find_depth_stencil_backup(depth_stencil);
	if (depth_stencil_backup == nullptr || depth_stencil_backup->backup_texture == 0) return;

	// Reset draw call stats for clears
	const draw_stats current_stats = stats.current;
	stats.current = { 0, 0 };

	// Ignore clears when the last viewport rendered to only affected a small subset of the depth-stencil (fixes flickering in some games)
	bool do_copy = true;
	switch (op)
	{
	case clear_op::clear_depth_stencil_view:
		// Mirror's Edge and Portal occasionally render something into a small viewport (16x16 in Mirror's Edge, 512x512 in Portal to render underwater geometry)
		do_copy = s_sunshine_auto ?
			check_aspect_ratio(current_stats.last_viewport.width, current_stats.last_viewport.height,
				static_cast<float>(depth_stencil_backup->frame_width), static_cast<float>(depth_stencil_backup->frame_height)) :
			(current_stats.last_viewport.width > 1024 || (current_stats.last_viewport.width == 0 || depth_stencil_backup->frame_width <= 1024));
		break;
	case clear_op::fullscreen_draw:
		// Mass Effect 3 in Mass Effect Legendary Edition sometimes uses a larger common depth buffer for shadow map and scene rendering, where the former uses a 1024x1024 viewport and the latter uses a viewport matching the render resolution
		do_copy = check_aspect_ratio(current_stats.last_viewport.width, current_stats.last_viewport.height, static_cast<float>(depth_stencil_backup->frame_width), static_cast<float>(depth_stencil_backup->frame_height));
		break;
	case clear_op::unbind_depth_stencil_view:
	case clear_op::direct_depth_switch:
		break;
	}

	if (do_copy)
	{
		if (op != clear_op::unbind_depth_stencil_view && op != clear_op::direct_depth_switch)
		{
			// If clear index override is set to zero, always copy any suitable buffers
			if (depth_stencil_backup->force_clear_index == 0)
			{
				// Use greater equals operator here to handle case where the same scene is first rendered into a shadow map and then for real (e.g. Mirror's Edge main menu)
				do_copy = current_stats.vertices >= stats.best_copy_stats.vertices || (op == clear_op::fullscreen_draw && current_stats.drawcalls >= stats.best_copy_stats.drawcalls);
			}
			else
			if (depth_stencil_backup->force_clear_index == std::numeric_limits<uint32_t>::max())
			{
				// Special case for Garry's Mod which chooses the last clear operation that has a high workload
				do_copy = current_stats.vertices >= 5000;
			}
			else
			{
				do_copy = (depth_stencil_backup->current_clear_index++) == (depth_stencil_backup->force_clear_index - 1);
			}

			if (demand.clear_history) stats.clears.push_back({ current_stats, op, do_copy });
		}

		// Make a backup copy of the depth texture before it is cleared
		if (do_copy)
		{
			stats.best_copy_stats = current_stats;

			stats.copied_during_frame = true;
			stats.copied_viewport = current_stats.last_viewport;
			stats.copy_provenance_ambiguous = false;
			stats.depth_copy = {};
			if (device->get_api() == device_api::d3d12)
			{
				if (state.is_queue) lock.unlock();
				auto ticket = sunshine_streamline::depth_capture::record_preserved(cmd_list->get_native(),
					depth_stencil.handle, D3D12_RESOURCE_STATE_DEPTH_WRITE);
				if (state.is_queue) lock.lock();
				stats.capture_ticket = std::move(ticket);
				stats.copied_during_frame = bool(stats.capture_ticket);
				if (stats.capture_ticket && s_streamline_probe_events)
				{
					sunshine_streamline::depth_resource_initialized(stats.capture_ticket.texture, stats.capture_ticket.resource_id,
						device->get_native(), true);
					stats.depth_copy = sunshine_streamline::record_depth_copy(cmd_list->get_native(),
						{depth_stencil.handle, depth_lifetime(device, depth_stencil)},
						{stats.capture_ticket.texture, stats.capture_ticket.resource_id});
				}
				stats.capture_marker = stats.depth_copy.recording;
				return;
			}
			stats.capture_marker = stats.depth_copy.recording;

			// Unlock before calling into device, since e.g. in D3D11 this can cause delayed destruction of resources (calls 'CDevice::FlushDeletionPool'), which calls 'on_destroy_resource' below, which tries to lock the same mutex
			if (state.is_queue)
				lock.unlock();

			const resource_desc desc = device->get_resource_desc(depth_stencil);
			if (desc.texture.samples > 1)
			{
				assert(device->check_capability(device_caps::resolve_depth_stencil) && (desc.usage & resource_usage::resolve_source) != 0);

				cmd_list->barrier(depth_stencil, resource_usage::depth_stencil_write, resource_usage::resolve_source);
				cmd_list->resolve_texture_region(depth_stencil, 0, nullptr, depth_stencil_backup->backup_texture, 0, 0, 0, 0, format_to_default_typed(desc.texture.format));
				cmd_list->barrier(depth_stencil, resource_usage::resolve_source, resource_usage::depth_stencil_write);
			}
			else
			{
				assert((desc.usage & resource_usage::copy_source) != 0);

				// A resource has to be in this state for a clear operation, so can assume it here
				cmd_list->barrier(depth_stencil, resource_usage::depth_stencil_write, resource_usage::copy_source);
				cmd_list->copy_resource(depth_stencil, depth_stencil_backup->backup_texture);
				cmd_list->barrier(depth_stencil, resource_usage::copy_source, resource_usage::depth_stencil_write);
			}
		}
	}
}

static void update_effect_runtime(effect_runtime *runtime)
{
	const auto &data = *runtime->get_private_data<generic_depth_data>();

	runtime->update_texture_bindings("DEPTH", data.selected_shader_resource, data.selected_shader_resource);

	sunshine_streamline::provider::set_depth_ready(runtime, data.selected_shader_resource != 0 && data.capture_ready);
}

static void on_reload_effect_runtime(effect_runtime *runtime)
{
	if (auto *data = runtime->get_private_data<generic_depth_data>(); data && data->native_driver)
	{
		// FX handles are recreated, but the native capture/scale owner is not.
		// Do not throw away a live capture epoch or retained real FG depth.
		sunshine_streamline::provider::reload_effect_bindings(runtime);
		if (!sunshine_streamline::provider::selected(runtime)) update_effect_runtime(runtime);
		return;
	}
	sunshine_streamline::provider::reload(runtime);
	if (auto *data = runtime->get_private_data<generic_depth_data>())
	{
		clear_raw_sample_request(*data);
		data->runtime_epoch = s_next_sample_scope.fetch_add(1, std::memory_order_relaxed);
		data->native_depth_requested = false;
		data->native_access_open = false;
		data->native_ui_state = native_depth_ui_state::inactive;
		data->native_ui_samples = 0;
		data->native_ui_present = 0;
		update_effect_runtime(runtime);
	}
}

static void on_init_device(device *device)
{
	device->create_private_data<generic_depth_device_data>();
	if (device->get_api() == device_api::d3d12)
	{
		auto *data = device->get_private_data<generic_depth_device_data>();
		data->native_identity = sunshine_native_identity::device_cookie(reinterpret_cast<ID3D12Device *>(device->get_native()), true);
		const std::unique_lock<std::shared_mutex> lock(s_mutex);
		if (data->native_identity) s_preservation_devices[data->native_identity] = data;
	}
	// Before the first DLSS feature is created, outside DLL loader lock. This
	// catches the game's linked NGX exports even if the runtime DLL loads later.
	if (device->get_api() == device_api::d3d12)
		sunshine_upscaler_trace::poll();
	reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "AutoSelectSceneDepth", s_sunshine_auto);

	reshade::get_config_value(nullptr, "DEPTH", "DisableINTZ", s_disable_intz);
	reshade::get_config_value(nullptr, "DEPTH", "DepthCopyBeforeClears", s_preserve_depth_buffers);
	unsigned int draw_heuristic = static_cast<unsigned int>(s_draw_stats_heuristic);
	unsigned int aspect_heuristic = static_cast<unsigned int>(s_aspect_ratio_heuristic);
	reshade::get_config_value(nullptr, "DEPTH", "DrawStatsHeuristic", draw_heuristic);
	reshade::get_config_value(nullptr, "DEPTH", "UseAspectRatioHeuristics", aspect_heuristic);
	s_draw_stats_heuristic = draw_heuristic <= static_cast<unsigned int>(draw_stats_heuristic::drawcalls) ?
		static_cast<draw_stats_heuristic>(draw_heuristic) : draw_stats_heuristic::prefer_vertices;
	s_aspect_ratio_heuristic = aspect_heuristic <= static_cast<unsigned int>(aspect_ratio_heuristic::match_custom_resolution_exactly) ?
		static_cast<aspect_ratio_heuristic>(aspect_heuristic) : aspect_ratio_heuristic::similar_aspect_ratio;

	reshade::get_config_value(nullptr, "DEPTH", "FilterFormat", s_format_filtering);
	reshade::get_config_value(nullptr, "DEPTH", "FilterResolutionWidth", s_custom_resolution_filtering[0]);
	reshade::get_config_value(nullptr, "DEPTH", "FilterResolutionHeight", s_custom_resolution_filtering[1]);

	if (s_aspect_ratio_heuristic > aspect_ratio_heuristic::match_custom_resolution_exactly)
		s_aspect_ratio_heuristic = aspect_ratio_heuristic::similar_aspect_ratio;
}
static void on_destroy_device(device *device)
{
	const generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();
	{
		const std::unique_lock<std::shared_mutex> lock(s_mutex);
		s_preservation_devices.erase(device_data->native_identity);
	}

	// Destroy any remaining resources
	for (const depth_stencil_backup &backup : device_data->depth_stencil_backups)
	{
		if (s_streamline_probe_events && device->get_api() == device_api::d3d12)
			sunshine_streamline::depth_resource_destroyed(backup.backup_texture.handle, backup.content_identity);
		device->destroy_resource(backup.backup_texture);
	}
	if (s_streamline_probe_events && device->get_api() == device_api::d3d12)
		for (const auto &[value, info] : device_data->depth_stencil_resources)
			sunshine_streamline::depth_resource_destroyed(value.handle, info.identity);

	device->destroy_private_data<generic_depth_device_data>();
}

static void on_init_command_list(command_list *cmd_list)
{
	cmd_list->create_private_data<state_tracking>(false);
	if (cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_capture::observe_command(cmd_list->get_native());
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
	{
		sunshine_streamline::command_initialized(cmd_list->get_native(), cmd_list->get_device()->get_native(),
			cmd_list->get_private_data<state_tracking>()->command_identity);
		sunshine_streamline::observe_native_discard_command(cmd_list->get_native());
	}
}
static void on_destroy_command_list(command_list *cmd_list)
{
	if (cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_capture::command_destroyed(cmd_list->get_native());
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::command_destroyed(cmd_list->get_native());
	cmd_list->destroy_private_data<state_tracking>();
}

static void on_init_command_queue(command_queue *cmd_queue)
{
	cmd_queue->create_private_data<state_tracking>(true);
	if (cmd_queue->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_capture::observe_queue(cmd_queue->get_native());
	if (s_streamline_probe_events && cmd_queue->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::queue_initialized(cmd_queue->get_native(), cmd_queue->get_device()->get_native(),
			cmd_queue->get_private_data<state_tracking>()->command_identity);

	if ((cmd_queue->get_type() & command_queue_type::graphics) == 0)
		return;

	generic_depth_device_data *const device_data = cmd_queue->get_device()->get_private_data<generic_depth_device_data>();
	device_data->queues.push_back(cmd_queue);
}
static void on_destroy_command_queue(command_queue *cmd_queue)
{
	if (cmd_queue->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_capture::retire_queue(cmd_queue->get_native());
	if (s_streamline_probe_events && cmd_queue->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::queue_destroyed(cmd_queue->get_native());
	cmd_queue->destroy_private_data<state_tracking>();

	device *const device = cmd_queue->get_device();
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();
	device_data->queues.erase(std::remove(device_data->queues.begin(), device_data->queues.end(), cmd_queue), device_data->queues.end());

	// All resources must be destroyed before a device reset in D3D9
	if (device->get_api() == device_api::d3d9)
	{
		for (const depth_stencil_backup &backup : device_data->depth_stencil_backups)
			device->destroy_resource(backup.backup_texture);
		device_data->depth_stencil_backups.clear();
	}
}

static void on_init_effect_runtime(effect_runtime *runtime)
{
	runtime->create_private_data<generic_depth_data>();
	set_generic_capture_enabled(runtime, *runtime->get_private_data<generic_depth_data>(), true);
	sunshine_streamline::provider::initialize(runtime);
}
static void on_destroy_effect_runtime(effect_runtime *runtime)
{
	auto &data = *runtime->get_private_data<generic_depth_data>();
	set_generic_capture_enabled(runtime, data, false);
	clear_raw_sample_request(data);
	sunshine_depth::retire_runtime(runtime);
	release_sample_reference(runtime, data);
	release_challenger(runtime, data, true);

	if (data.selected_shader_resource != 0 && !owns_member_view(data, data.selected_shader_resource))
	{
		device *const device = runtime->get_device();

		device->destroy_resource_view(data.selected_shader_resource);

		generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();
		device_data->untrack_depth_stencil(device, data.selected_depth_stencil);
	}
	if (auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>())
	{
		retire_provided_members(runtime, data, *device_data, true);
		retire_member_views(runtime, data, *device_data, true);
	}

	runtime->destroy_private_data<generic_depth_data>();
	sunshine_streamline::provider::destroy(runtime);
}

static bool on_create_resource(device *device, resource_desc &desc, subresource_data *, resource_usage)
{
	if (desc.type != resource_type::surface && desc.type != resource_type::texture_2d)
		return false; // Skip resources that are not 2D textures

	if ((desc.texture.samples > 1 && !device->check_capability(device_caps::resolve_depth_stencil)) || (desc.usage & resource_usage::depth_stencil) == 0 || desc.texture.format == format::s8_uint)
		return false; // Skip multisampled textures and resources that are not used as depth buffers

	switch (device->get_api())
	{
	case device_api::d3d9:
		if (s_disable_intz || desc.texture.samples > 1)
			return false;
		// Skip textures that are sampled as PCF shadow maps (see https://aras-p.info/texts/D3D9GPUHacks.html#shadowmap) using hardware support, since changing format would break that
		if (desc.type == resource_type::texture_2d && (desc.texture.format == format::d16_unorm || desc.texture.format == format::d24_unorm_x8_uint || desc.texture.format == format::d24_unorm_s8_uint))
			return false;
		// Skip small textures that are likely just shadow maps too (fixes a hang in Dragon's Dogma: Dark Arisen when changing areas)
		if (desc.texture.width <= 512)
			return false;
		if (desc.texture.format == format::d32_float || desc.texture.format == format::d32_float_s8_uint)
			reshade::log::message(reshade::log::level::warning, "Replacing high bit depth depth-stencil format with a lower bit depth format");
		// Replace texture format with special format that supports normal sampling (see https://aras-p.info/texts/D3D9GPUHacks.html#depth)
		desc.texture.format = format::intz;
		desc.usage |= resource_usage::shader_resource;
		break;
	case device_api::d3d10:
	case device_api::d3d11:
		// Allow shader access to textures that are used as depth-stencil attachments
		desc.texture.format = format_to_typeless(desc.texture.format);
		desc.usage |= resource_usage::shader_resource;
		break;
	case device_api::d3d12:
	case device_api::vulkan:
		// D3D12 and Vulkan always use backup texture, but need to be able to copy to it
		if (desc.texture.samples > 1)
			desc.usage |= resource_usage::resolve_source;
		else
			desc.usage |= resource_usage::copy_source;
		break;
	case device_api::opengl:
		// No need to change anything in OpenGL
		return false;
	}

	return true;
}
static bool on_create_resource_view(device *device, resource resource, resource_usage usage_type, resource_view_desc &desc)
{
	// A view cannot be created with a typeless format (which was set in 'on_create_resource' above), so fix it in case defaults are used
	if ((device->get_api() != device_api::d3d10 && device->get_api() != device_api::d3d11) || (desc.format != format::unknown && desc.format != format_to_typeless(desc.format)))
		return false;

	const resource_desc texture_desc = device->get_resource_desc(resource);
	// Only non-multisampled textures where modified, so skip all others
	if ((texture_desc.texture.samples > 1 && !device->check_capability(device_caps::resolve_depth_stencil)) || (texture_desc.usage & resource_usage::depth_stencil) == 0)
		return false;

	switch (usage_type)
	{
	case resource_usage::depth_stencil:
		desc.format = format_to_depth_stencil_typed(texture_desc.texture.format);
		break;
	case resource_usage::shader_resource:
		desc.format = format_to_default_typed(texture_desc.texture.format);
		break;
	}

	// Only need to set the rest of the members if the application did not pass in a valid description already
	if (desc.type == resource_view_type::unknown)
	{
		desc.type = texture_desc.texture.depth_or_layers > 1 ? resource_view_type::texture_2d_array : resource_view_type::texture_2d;
		desc.texture.first_level = 0;
		desc.texture.levels = (usage_type == resource_usage::shader_resource) ? UINT32_MAX : 1;
		desc.texture.first_layer = 0;
		desc.texture.layers = (usage_type == resource_usage::shader_resource) ? UINT32_MAX : 1;
	}

	return true;
}
static void on_init_resource(device *device, const resource_desc &desc, const subresource_data *, resource_usage, resource resource)
{
	if (desc.type != resource_type::surface && desc.type != resource_type::texture_2d && (desc.type != resource_type::texture_3d || desc.texture.depth_or_layers > 1))
		return;

	if ((desc.usage & resource_usage::depth_stencil) == 0)
		return;

	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();

	const std::unique_lock<std::shared_mutex> lock(s_mutex);

	depth_stencil_resource info;
	info.desc = desc;
	info.identity = sunshine_native_identity::allocate_identity();
	if (device->get_api() == device_api::d3d12)
	{
		// Native object identity is shared with SL/NGX. No pointer or ID is
		// persisted after destruction or saved in the user's configuration.
		const auto native = sunshine_native_identity::identify_resource(resource.handle);
		if (native && native.device == device_data->native_identity) { info.identity = native.resource; info.api_identity = true; }
	}
	if (s_streamline_probe_events && device->get_api() == device_api::d3d12)
		sunshine_streamline::depth_resource_initialized(resource.handle, info.identity, device->get_native(), content_supported(desc));
	device_data->depth_stencil_resources.emplace(resource, std::move(info));
}
static void on_destroy_resource(device *device, resource resource)
{
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();

	// In some cases the 'destroy_device' event may be called before all resources have been destroyed
	// The state tracking context would have been destroyed already in that case, so return early if it does not exist
	if (device_data == nullptr)
		return;

	std::unique_lock<std::shared_mutex> lock(s_mutex);

	// Remove this destroyed resource from the list of tracked depth-stencil resources
	if (const auto it = device_data->depth_stencil_resources.find(resource);
		it != device_data->depth_stencil_resources.end())
	{
		if (s_streamline_probe_events && device->get_api() == device_api::d3d12)
			sunshine_streamline::depth_resource_destroyed(resource.handle, it->second.identity);
		device_data->depth_stencil_resources.erase(it);

		// A backup resource is always created in D3D12 and Vulkan, so to find out if an effect runtime references this depth-stencil resource, can simply check if a backup resource was created for it
		if (device_data->find_depth_stencil_backup(resource) != nullptr)
		{
			lock.unlock();

			reshade::log::message(reshade::log::level::warning, "A depth-stencil resource was destroyed while still in use.");

			// This is bad ... the resource may still be in use by an effect on the GPU and destroying it would crash it
			// Try to mitigate that somehow by delaying this thread a little to hopefully give the GPU enough time to catch up before the resource memory is deallocated
			Sleep(500);
		}
	}
}

static bool on_draw(command_list *cmd_list, uint32_t vertices, uint32_t instances, uint32_t, uint32_t)
{
	if (!vertices || !instances) return false;
	auto &state = *cmd_list->get_private_data<state_tracking>();
	if (state.current_depth_stencil == 0)
		return false; // This is a draw call with no depth-stencil bound

	// Check if this draw call likely represets a fullscreen rectangle (two triangles), which would clear the depth-stencil
	const bool fullscreen_draw = vertices == 6 && instances == 1;
	if (fullscreen_draw &&
		s_preserve_depth_buffers == 2 &&
		state.first_draw_since_bind &&
		// But ignore that in Vulkan (since it is invalid to copy a resource inside an active render pass)
		cmd_list->get_device()->get_api() != device_api::vulkan)
		on_clear_depth_impl(cmd_list, state, state.current_depth_stencil, clear_op::fullscreen_draw);
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12 && state.current_depth_lifetime)
		sunshine_streamline::depth_content_write(cmd_list->get_native(), state.current_depth_stencil.handle, state.current_depth_lifetime);

	// If this is queue state (happens if this is a immediate command list), need to protect access to it, since another thread may be in a present call, which can reset it
	std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
	if (state.is_queue)
		lock.lock();

	state.first_draw_since_bind = false;

	depth_stencil_frame_stats &stats = state.stats_per_used_depth_stencil[state.current_depth_stencil];
	const auto demand = tracking_demand(cmd_list->get_device());
	stats.limit_tracking(demand);
	stats.total.observed_work = stats.current.observed_work = true;
	if (demand.counters)
	{
		stats.total.vertices += vertices * instances;
		stats.total.drawcalls += 1;
		stats.current.vertices += vertices * instances;
		stats.current.drawcalls += 1;
	}
	if (state.current_viewport.width * state.current_viewport.height >= stats.total.last_viewport.width * stats.total.last_viewport.height)
		stats.total.last_viewport = state.current_viewport;

	// Skip updating last viewport for fullscreen draw calls, to prevent a clear operation in Prince of Persia: The Sands of Time from getting filtered out
	if (!fullscreen_draw)
		stats.current.last_viewport = state.current_viewport;

	return false;
}
static bool on_draw_indexed(command_list *cmd_list, uint32_t indices, uint32_t instances, uint32_t, int32_t, uint32_t)
{
	return on_draw(cmd_list, indices, instances, 0, 0);
}
static bool on_draw_indirect(command_list *cmd_list, indirect_command type, resource, uint64_t, uint32_t draw_count, uint32_t)
{
	if (!draw_count || type == indirect_command::dispatch || type == indirect_command::dispatch_rays)
		return false; // Compute/ray dispatches do not rasterize into the bound DSV.
	// ReShade 6.8 reports D3D12 ExecuteIndirect as 'unknown' even for raster draws.
	// Preserve that existing activity evidence until the backend exposes its signature.

	auto &state = *cmd_list->get_private_data<state_tracking>();
	if (state.current_depth_stencil == 0)
		return false; // This is a draw call with no depth-stencil bound
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12 && state.current_depth_lifetime)
		sunshine_streamline::depth_content_write(cmd_list->get_native(), state.current_depth_stencil.handle, state.current_depth_lifetime);

	// If this is queue state (happens if this is a immediate command list), need to protect access to it, since another thread may be in a present call, which can reset it
	std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
	if (state.is_queue)
		lock.lock();

	state.first_draw_since_bind = false;
	depth_stencil_frame_stats &stats = state.stats_per_used_depth_stencil[state.current_depth_stencil];
	const auto demand = tracking_demand(cmd_list->get_device());
	stats.limit_tracking(demand);
	stats.total.observed_work = stats.current.observed_work = true;
	if (demand.counters)
	{
		stats.total.drawcalls += draw_count;
		stats.total.drawcalls_indirect += draw_count;
		stats.current.drawcalls += draw_count;
		stats.current.drawcalls_indirect += draw_count;
	}
	stats.current.last_viewport = state.current_viewport;
	if (state.current_viewport.width * state.current_viewport.height >= stats.total.last_viewport.width * stats.total.last_viewport.height)
		stats.total.last_viewport = state.current_viewport;

	return false;
}

static void on_bind_viewport(command_list *cmd_list, uint32_t first, uint32_t count, const viewport *viewport)
{
	if (first != 0 || count == 0)
		return; // Only interested in the main viewport

	auto &state = *cmd_list->get_private_data<state_tracking>();
	state.current_viewport = viewport[0];
}
static void on_bind_depth_stencil_impl(command_list *cmd_list, resource_view depth_stencil_view, bool direct_binding)
{
	auto &state = *cmd_list->get_private_data<state_tracking>();

	const resource depth_stencil = (depth_stencil_view != 0) ? cmd_list->get_device()->get_resource_from_view(depth_stencil_view) : resource { 0 };

	if (depth_stencil != state.current_depth_stencil)
	{
		if (depth_stencil != 0)
			state.first_draw_since_bind = true;

		const auto api = cmd_list->get_device()->get_api();
		const auto previous = state.stats_per_used_depth_stencil.find(state.current_depth_stencil);
		// An actual D3D12 OMSetRenderTargets A->B is also an unbind of A. Only
		// extend that boundary when this list still proves A is DEPTH_WRITE.
		// Synthetic render-pass bindings and other APIs retain their old behavior.
		const bool proven_direct_switch = direct_binding && api == device_api::d3d12 &&
			depth_stencil != 0 && !state.inside_render_pass &&
			previous != state.stats_per_used_depth_stencil.end() && previous->second.direct_clear_depth_write;
		// Make a backup before the depth is used differently. Preserve the
		// established A->null path unchanged; do not enable end-of-frame access.
		if (s_preserve_depth_buffers == 2 &&
			state.current_depth_stencil != 0 &&
			((depth_stencil == 0 && (api == device_api::d3d12 || api == device_api::vulkan)) || proven_direct_switch))
			on_clear_depth_impl(cmd_list, state, state.current_depth_stencil,
				proven_direct_switch ? clear_op::direct_depth_switch : clear_op::unbind_depth_stencil_view);
	}

	state.current_depth_stencil = depth_stencil;
	if (cmd_list->get_device()->get_api() == device_api::d3d12)
		state.current_depth_lifetime = s_streamline_probe_events ? depth_lifetime(cmd_list->get_device(), depth_stencil) : 0;
}
static void on_bind_depth_stencil(command_list *cmd_list, uint32_t, const resource_view *, resource_view depth_stencil_view)
{
	on_bind_depth_stencil_impl(cmd_list, depth_stencil_view, true);
}
static bool on_clear_depth_stencil(command_list *cmd_list, resource_view dsv, const float *depth, const uint8_t *, uint32_t, const rect *)
{
	// Ignore clears that do not affect the depth buffer (stencil clears)
	if (dsv != 0 && depth != nullptr)
	{
		auto &state = *cmd_list->get_private_data<state_tracking>();

		const resource depth_stencil = cmd_list->get_device()->get_resource_from_view(dsv);

		// Note: This does not work when called from 'vkCmdClearAttachments', since it is invalid to copy a resource inside an active render pass
		if (s_preserve_depth_buffers)
			on_clear_depth_impl(cmd_list, state, depth_stencil, clear_op::clear_depth_stencil_view);
		// The preserved snapshot precedes the game clear, which changes the source.
		if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
			sunshine_streamline::depth_content_write(cmd_list->get_native(), depth_stencil.handle);

		std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
		if (state.is_queue)
			lock.lock();
		auto &stats = state.stats_per_used_depth_stencil[depth_stencil];
		stats.limit_tracking(tracking_demand(cmd_list->get_device()));
		if (cmd_list->get_device()->get_api() == device_api::d3d12 && !state.inside_render_pass)
			stats.direct_clear_depth_write = true;
		stats.reversed_clear_value |= *depth != 1.0f; // Preserve the inherited UI hint.
		stats.clear_zero |= *depth == 0.0f;
		stats.clear_one |= *depth == 1.0f;
		stats.clear_other |= *depth != 0.0f && *depth != 1.0f;
	}

	return false;
}
static bool on_begin_render_pass_with_depth_stencil(command_list *cmd_list, uint32_t, const render_pass_render_target_desc *, const render_pass_depth_stencil_desc *depth_stencil_desc, render_pass_flags)
{
	auto &render_state = *cmd_list->get_private_data<state_tracking>();
	// Load/store actions do not expose a safe ordinary OM binding boundary.
	// Mark this before synthesizing the clear/bind notifications below.
	render_state.invalidate_direct_clear_proofs();
	render_state.inside_render_pass = true;
	if (s_streamline_probe_events)
	{
		auto &pass_state = *cmd_list->get_private_data<state_tracking>();
		pass_state.render_pass_depth = depth_stencil_desc != nullptr && depth_stencil_desc->view != 0 ?
			cmd_list->get_device()->get_resource_from_view(depth_stencil_desc->view) : resource {};
		// The generic API loses native ending-resolve details. Retain that boundary
		// as incomplete evidence rather than assuming that store means preservation.
		pass_state.render_pass_depth_uncertain = pass_state.render_pass_depth != 0;
		if (cmd_list->get_device()->get_api() == device_api::d3d12 && pass_state.render_pass_depth != 0 &&
			depth_stencil_desc->depth_load_op == render_pass_load_op::discard)
			sunshine_streamline::depth_content_invalidate(cmd_list->get_native(), pass_state.render_pass_depth.handle);
	}
	if (depth_stencil_desc != nullptr && depth_stencil_desc->depth_load_op == render_pass_load_op::clear)
	{
		on_clear_depth_stencil(cmd_list, depth_stencil_desc->view, &depth_stencil_desc->clear_depth, nullptr, 0, nullptr);

		// Prevent 'on_bind_depth_stencil' from copying depth buffer again
		auto &state = *cmd_list->get_private_data<state_tracking>();
		state.current_depth_stencil = { 0 };
	}

	// If render pass has depth store operation set to 'discard', any copy performed after the render pass will likely contain broken data, so can only hope that the depth buffer can be copied before that ...

	on_bind_depth_stencil_impl(cmd_list, depth_stencil_desc != nullptr ? depth_stencil_desc->view : resource_view {}, false);
	return false;
}

static bool on_end_render_pass(command_list *cmd_list)
{
	auto &state = *cmd_list->get_private_data<state_tracking>();
	state.inside_render_pass = false;
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12 && state.render_pass_depth_uncertain)
		sunshine_streamline::depth_content_invalidate(cmd_list->get_native(), state.render_pass_depth.handle);
	state.render_pass_depth = {};
	state.render_pass_depth_uncertain = false;
	return false;
}
// Call under the queue-state lock when needed. These writes establish activity,
// not useful contents or a vertex estimate; the existing histogram still qualifies them.
static void record_scene_write(state_tracking &state, resource target, const viewport &view, depth_tracking_demand demand)
{
	if (target == 0) return;
	auto &stats = state.stats_per_used_depth_stencil[target];
	stats.limit_tracking(demand);
	stats.total.observed_work = stats.current.observed_work = true;
	if (demand.counters)
	{
		++stats.total.scene_writes;
		++stats.current.scene_writes;
	}
	stats.current.last_viewport = view;
	if (view.width * view.height >= stats.total.last_viewport.width * stats.total.last_viewport.height)
		stats.total.last_viewport = view;
	if (target == state.current_depth_stencil) state.first_draw_since_bind = false;
}

static void record_copied_depth(command_list *cmd_list, resource dest, uint32_t subresource = 0)
{
	if (dest == 0 || subresource != 0) return; // The shader samples the first depth plane/level/layer.
	auto *device_data = cmd_list->get_device()->get_private_data<generic_depth_device_data>();
	if (device_data == nullptr) return;
	auto &state = *cmd_list->get_private_data<state_tracking>();
	const std::shared_lock<std::shared_mutex> lock(s_mutex);
	const auto found = device_data->depth_stencil_resources.find(dest);
	if (found == device_data->depth_stencil_resources.end()) return; // Skip color and our sampling backups.
	// A transfer is activity, not proof of a full raster viewport. In particular,
	// ReShade can report a null destination box for a partial origin copy.
	record_scene_write(state, dest, {}, tracking_demand(*device_data));
}

static bool on_dispatch_mesh(command_list *cmd_list, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z)
{
	auto &state = *cmd_list->get_private_data<state_tracking>();
	if (!groups_x || !groups_y || !groups_z || state.current_depth_stencil == 0) return false;
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12 && state.current_depth_lifetime)
		sunshine_streamline::depth_content_write(cmd_list->get_native(), state.current_depth_stencil.handle, state.current_depth_lifetime);
	std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
	if (state.is_queue) lock.lock();
	record_scene_write(state, state.current_depth_stencil, state.current_viewport, tracking_demand(cmd_list->get_device()));
	return false;
}
static bool on_copy_resource(command_list *cmd_list, resource, resource dest)
{
	record_copied_depth(cmd_list, dest);
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_content_write(cmd_list->get_native(), dest.handle);
	return false;
}
static bool empty_copy_box(const subresource_box *box)
{
	return box != nullptr && (box->left >= box->right || box->top >= box->bottom || box->front >= box->back);
}
static bool on_copy_texture_region(command_list *cmd_list, resource, uint32_t, const subresource_box *source_box,
	resource dest, uint32_t subresource, const subresource_box *dest_box, filter_mode)
{
	if (empty_copy_box(source_box) || empty_copy_box(dest_box)) return false;
	record_copied_depth(cmd_list, dest, subresource);
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_content_invalidate(cmd_list->get_native(), dest.handle);
	return false;
}
static bool on_copy_buffer_to_texture(command_list *cmd_list, resource, uint64_t, uint32_t, uint32_t,
	resource dest, uint32_t subresource, const subresource_box *dest_box)
{
	if (empty_copy_box(dest_box)) return false;
	record_copied_depth(cmd_list, dest, subresource);
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_content_invalidate(cmd_list->get_native(), dest.handle);
	return false;
}
static bool on_resolve_texture_region(command_list *cmd_list, resource, uint32_t, const subresource_box *source_box,
	resource dest, uint32_t subresource, uint32_t, uint32_t, uint32_t, format)
{
	if (empty_copy_box(source_box)) return false;
	record_copied_depth(cmd_list, dest, subresource);
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_content_invalidate(cmd_list->get_native(), dest.handle);
	return false;
}
static void on_content_barrier(command_list *cmd_list, uint32_t count, const resource *resources,
	const resource_usage *old_states, const resource_usage *new_states)
{
	if (cmd_list->get_device()->get_api() != device_api::d3d12) return;
	auto &state = *cmd_list->get_private_data<state_tracking>();
	{
		std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
		if (state.is_queue) lock.lock();
		for (uint32_t i = 0; i != count; ++i)
		{
			// Alias/global/unknown barriers may affect a previous resource which
			// the generic callback does not identify. Never derive proof from the
			// reported new state (enhanced-barrier layouts are not represented).
			if (resources[i] == 0 || old_states[i] == resource_usage::undefined || new_states[i] == resource_usage::undefined)
			{
				state.invalidate_direct_clear_proofs();
				break;
			}
			if (const auto found = state.stats_per_used_depth_stencil.find(resources[i]); found != state.stats_per_used_depth_stencil.end())
				found->second.direct_clear_depth_write = false;
		}
	}
	if (!s_streamline_probe_events) return;
	for (uint32_t i = 0; i != count; ++i)
	{
		// ReShade's alias callback retains only the resource after the alias.
		// The previous resource can be our depth or backup even when this handle
		// is unrelated, so its undefined old state invalidates all prior evidence.
		if (resources[i] == 0 || old_states[i] == resource_usage::undefined)
			sunshine_streamline::depth_content_invalidate(cmd_list->get_native());
		else if (new_states[i] == resource_usage::undefined ||
			(old_states[i] & resource_usage::unordered_access) != 0 || (new_states[i] & resource_usage::unordered_access) != 0)
			sunshine_streamline::depth_content_invalidate(cmd_list->get_native(), resources[i].handle);
	}
}

static void on_reset(command_list *cmd_list)
{
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
	{
		sunshine_streamline::command_reset(cmd_list->get_native(), cmd_list->get_device()->get_native(),
			cmd_list->get_private_data<state_tracking>()->command_identity);
		sunshine_streamline::observe_native_discard_command(cmd_list->get_native());
	}
	auto &target_state = *cmd_list->get_private_data<state_tracking>();
	target_state.reset();
}
static void on_close(command_list *cmd_list)
{
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::command_closed(cmd_list->get_native());
}
static void on_execute_primary(command_queue *queue, command_list *cmd_list)
{
	if (s_streamline_probe_events && queue->get_device()->get_api() == device_api::d3d12)
	{
		// A missed lifecycle callback revokes old evidence. The independently
		// owned lifetime cookie lets the next reset/execute recover live objects.
		sunshine_streamline::queue_initialized(queue->get_native(), queue->get_device()->get_native(),
			queue->get_private_data<state_tracking>()->command_identity);
		sunshine_streamline::command_executed(queue->get_native(), cmd_list->get_native());
	}
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
	if (s_streamline_probe_events && cmd_list->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::command_secondary_executed(cmd_list->get_native(), secondary_cmd_list->get_native());
	auto &target_state = *cmd_list->get_private_data<state_tracking>();
	const auto &source_state = *secondary_cmd_list->get_private_data<state_tracking>();
	{
		std::shared_lock<std::shared_mutex> lock(s_mutex, std::defer_lock);
		if (target_state.is_queue) lock.lock();
		target_state.invalidate_direct_clear_proofs();
	}

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

static void on_present(command_queue *presenting_queue, swapchain *swapchain, const rect *, const rect *, uint32_t, const rect *)
{
	// The native tag's UntilPresent lifetime ends here. Already copied packets
	// remain owned and usable by effects; tags supplied after this boundary keep
	// their new authority, including concurrent next-frame tagging.
	if (presenting_queue->get_device()->get_api() == device_api::d3d12)
		sunshine_streamline::depth_capture::present(presenting_queue->get_native());
	sunshine_depth::begin_present(swapchain);
	device *const device = swapchain->get_device();
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();

	// Declared before the lock, so its destructor emits only after unlocking.
	std::optional<activity_present_trace> trace;
	if (s_frame_activity_trace) trace.emplace(swapchain, presenting_queue);
	const std::unique_lock<std::shared_mutex> lock(s_mutex);
	++device_data->native_present_index;
	consume_depth_candidate_list_request(*device_data);
	const auto demand = tracking_demand(*device_data);
	if (trace) trace->begin(*device_data);

	state_tracking queue_state(true);
	// Merge state from all graphics queues
	for (command_queue *const queue : device_data->queues)
	{
		auto &state = *queue->get_private_data<state_tracking>();
		for (auto &[depth_stencil, stats] : state.stats_per_used_depth_stencil)
		{
			if (const auto earlier_queue = queue_state.stats_per_used_depth_stencil.find(depth_stencil);
				stats.copied_during_frame && earlier_queue != queue_state.stats_per_used_depth_stencil.end() && earlier_queue->second.copied_during_frame)
				stats.copy_provenance_ambiguous = true; // Queue iteration is not GPU execution order.
		}
		if (trace) trace->add(queue, state, *device_data);
		queue_state.merge(state, demand);

		state.reset_on_present();
	}

	// Only update device list if there are any depth-stencils, otherwise this may be a second present call (at which point 'reset_on_present' already cleared out the queue list in the first present call)
	if (queue_state.stats_per_used_depth_stencil.empty())
		return;

	// Also skip update when there has been very little activity (special case for emulators like PCSX2 which may present more often than they render a frame)
	if (!s_sunshine_auto && queue_state.stats_per_used_depth_stencil.size() == 1 && queue_state.stats_per_used_depth_stencil.begin()->second.total.drawcalls != 0 &&
		queue_state.stats_per_used_depth_stencil.begin()->second.total.drawcalls <= 8 && queue_state.stats_per_used_depth_stencil.begin()->second.total.scene_writes == 0)
		return;

	device_data->frame_index++;
	device_data->native_depth_present_index = device_data->native_present_index;
	if (trace)
	{
		trace->frame_after = device_data->frame_index;
		trace->depth_present = device_data->native_depth_present_index;
	}

	for (const auto &[depth_stencil, frame_stats] : queue_state.stats_per_used_depth_stencil)
	{
		if (const auto it = device_data->depth_stencil_resources.find(depth_stencil);
			it != device_data->depth_stencil_resources.end()) // Otherwise resource was destroyed
		{
			depth_stencil_resource &info = it->second;
			observe_preserved_layout(info, frame_stats);
			info.orientation_evidence.observe(device_data->frame_index, frame_stats.clear_zero, frame_stats.clear_one, frame_stats.clear_other);

			// Save to current list of depth-stencils on the device
			info.last_frame_stats = frame_stats;
			info.last_used_in_frame = device_data->frame_index;

			if (std::numeric_limits<uint64_t>::max() == info.first_used_in_frame)
				info.first_used_in_frame = device_data->frame_index;
		}
	}

	// Destroy resources that were enqueued for delayed destruction and have reached the targeted number of passed frames
	for (auto it = device_data->depth_stencil_backups.begin(); it != device_data->depth_stencil_backups.end();)
	{
		if (device_data->frame_index >= it->destroy_after_frame)
		{
			assert(it->references == 0);

			device->destroy_resource(it->backup_texture);
			it = device_data->depth_stencil_backups.erase(it);
			continue;
		}

		// Reset current clear index
		it->current_clear_index = 0;

		++it;
	}
}

static void prepare_probe_target(effect_runtime *runtime, command_list *cmd_list, generic_depth_data &data,
	generic_depth_device_data &device_data, uint32_t frame_width, uint32_t frame_height)
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
	if ((data.override_depth_stencil != 0 && !data.manual_recovery_probing) || (!s_sunshine_auto && data.native_depth_requested))
	{
		// Calibration remains available with content-based selection disabled.
		// Sample only the selected backup; never allocate challengers or switch it.
		data.probe_target_id = data.selected_identity;
		return;
	}
	if (!s_sunshine_auto)
		return;

	if (data.challenger_depth_stencil == data.selected_depth_stencil && data.challenger_depth_stencil != 0)
		release_challenger(runtime, data); // The selected view now holds its own tracking reference.

	if (sunshine_depth::busy() || GetTickCount64() < data.next_sample_tick)
		return;

	if (data.probe_target_id != 0 && sunshine_depth_probe::phase_expired(data.probe_started_tick, GetTickCount64()))
		release_challenger(runtime, data); // Missing/unsupported capture must not pin the round robin.
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
		backup->frame_width = frame_width;
		backup->frame_height = frame_height;
		if (s_preserve_depth_buffers)
			reshade::get_config_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", backup->force_clear_index);
		data.challenger_depth_stencil = target;
		data.challenger_id = info.identity;
		data.challenger_started_frame = device_data.frame_index;
		// Retirement wait is not measurement time. Both the queued visit and
		// this allocated capture phase have their own bounded two-second clock.
		data.probe_started_tick = GetTickCount64();
		data.challenger_samples = 0;
		return; // Let the ordinary clear/unbind hooks see one full rendered frame.
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
		// Before-clear D3D12 pixels now live in the common owner's snapshot.
		// Unqualified challengers have no member/selected publication path yet;
		// publish their existing snapshot so the probe can establish quality.
		if (device->get_api() == device_api::d3d12)
			preserve_depth_at_effects(runtime, cmd_list, device_data, target, info.identity, data.challenger_started_frame);
		return;
	}

	// Preserve the probe's historical mode-2 policy on every backend.
	if (s_preserve_depth_buffers != 2 && (info.desc.usage & (resource_usage::copy_source | resource_usage::resolve_source)) != 0)
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
	generic_depth_device_data &device_data, const sunshine_depth::sample_result &completed,
	const sunshine_depth::depth_quality &quality)
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
		quality.kind != sunshine_depth::content_kind::useful)
		return; // Only authenticated useful scene captures become numeric evidence.
	data.raw_latest_submission = associated.captured;
	data.raw_latest_sample = {};
	data.raw_latest_sample.id = associated.sample.id;
	data.raw_latest_sample.capture_ms = associated.sample.capture_ms;
	data.raw_latest_sample.metadata = data.raw_pending_metadata;
	data.raw_latest_sample.readback_frame = data.raw_pending_metadata.frame;
	data.raw_latest_sample.readback_source = data.raw_pending_metadata.source;
	data.raw_latest_sample.readback_layout_epoch = associated.captured.selection.layout_epoch;
	data.raw_latest_sample.raw = associated.sample.raw;
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
	if ((!s_sunshine_auto && data.override_depth_stencil == 0 && !data.native_depth_requested && !data.raw_scene_requested) || target_id == 0 ||
		sunshine_depth::busy() || (!raw_selected_due && now < data.next_sample_tick))
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
		binding.camera.unit_epoch = data.raw_basis_epoch;
		binding.camera.frame = { captured.record.frame, captured.record.runtime };
		binding.camera.source = binding.selection.original;
		binding.projection_associated = false;
		const auto &copy = captured.metadata.depth_copy;
		binding.copy = sunshine_camera_binding::observed_copy_key(copy);
		binding.capture_frame = captured.record.frame;
		binding.capture_ms = now;
		if (sunshine_streamline::enabled())
		{
			// This is the actual submitted backup, not a subsequently selected
			// source or a camera fetched when its asynchronous readback arrives.
			auto frame = captured.metadata;
			if (device->get_api() == device_api::d3d12) frame.command_queue = runtime->get_command_queue()->get_native();
			const auto selected = sunshine_depth::camera_selection_snapshot(frame);
			sunshine_streamline::evaluation_snapshot evidence;
			sunshine_streamline::query_evaluation(selected, evidence);
			sunshine_camera_binding::capture_camera_observation(binding, selected, evidence);
		}
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
		data.next_sample_tick = now + (data.raw_scene_requested ? 125 : ((data.override_depth_stencil != 0 && !data.manual_recovery_probing) || !s_sunshine_auto ? 1000 : 250));
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

static void begin_depth_frame(effect_runtime *runtime, command_list *cmd_list)
{
	device *const device = runtime->get_device();
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();

	if (device_data == nullptr)
		return;

	auto &data = *runtime->get_private_data<generic_depth_data>();
	const bool automatic = s_sunshine_auto && (device->get_api() == device_api::d3d11 || device->get_api() == device_api::d3d12);
	data.native_access_open = false;
	auto completed_sample = sunshine_depth::poll(runtime);
	if (completed_sample && sunshine_streamline::provider::complete(runtime, *completed_sample))
		completed_sample.reset();
	// One content classification serves source preference and numeric evidence.
	// Flat interior loading frames must not establish H before a scene appears.
	const auto completed_quality = completed_sample && completed_sample->valid ?
		sunshine_depth::analyze_depth(completed_sample->values.data(), completed_sample->values.size(), completed_sample->width, completed_sample->height) :
		sunshine_depth::depth_quality {};
	const bool completed_other_raw_sample = completed_sample && completed_sample->token != data.challenger_id &&
		completed_sample->capture_id != 0 && completed_sample->capture_id == data.raw_pending_capture_id && data.raw_pending_metadata.depth_ready;
	if (completed_sample)
	{
		complete_raw_sample(runtime, data, *device_data, *completed_sample, completed_quality);
		release_sample_reference(runtime, data);
	}
	const bool was_streamline = sunshine_streamline::provider::selected(runtime);
	retire_provided_members(runtime, data, *device_data);
	const bool api_selected = sunshine_streamline::provider::begin(runtime, cmd_list, device_data->native_present_index,
		automatic && data.override_depth_stencil == 0 && s_streamline_source_events);
	// A native API snapshot needs no Generic work. Shared preservation enables
	// itself at its first copy above; keep that demand through temporary gaps.
	// Manual pins and fallback resume capture before entering the selector.
	set_generic_capture_enabled(runtime, data,
		!api_selected || sunshine_streamline::provider::uses_shared_preservation(runtime));
	if (api_selected)
	{
		if (!was_streamline) release_challenger(runtime, data, true);
		data.native_access_present = device_data->native_present_index;
		data.native_access_open = true;
		return;
	}
	if (was_streamline) update_effect_runtime(runtime);

	resource selected_depth_stencil = { 0 };
	const depth_selection_resource *selected_depth_stencil_info = nullptr;

	uint32_t frame_width, frame_height;
	runtime->get_screenshot_width_and_height(&frame_width, &frame_height);

	std::shared_lock<std::shared_mutex> lock(s_mutex);
	const uint64_t native_present_index = device_data->native_present_index;
	if (s_frame_activity_trace)
	{
		data.trace_snapshot_present = native_present_index;
		data.trace_snapshot_frame = device_data->frame_index;
	}
	const depth_selection_snapshot current_depth_stencil_resources = snapshot_depth_selection(device_data->depth_stencil_resources);
	// Unlock before calling into device below, since device may hold a lock itself and that then can deadlock another thread that calls into 'on_destroy_resource' from the device holding that lock
	lock.unlock();

	uint64_t current_id = 0;
	const auto policy_source = data.anchor_identity ? data.anchor_depth_stencil : data.selected_depth_stencil;
	const auto policy_identity = data.anchor_identity ? data.anchor_identity : data.selected_identity;
	if (const auto current = find_depth_selection(current_depth_stencil_resources, policy_source);
		current != current_depth_stencil_resources.end() && current->second.identity == policy_identity)
		current_id = current->second.identity;

	std::vector<uint64_t> eligible_ids;
	bool any_recent_activity = false;
	bool any_filtered = false;
	bool any_warming = false;
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

		if ((s_format_filtering != 0 && !check_depth_format(info.desc.texture.format)) || !candidate_matches_shape(info, frame_width, frame_height))
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
	data.manual_recovery_probing = false;
	if (data.override_depth_stencil != 0)
	{
		const auto manual = find_depth_selection(current_depth_stencil_resources, data.override_depth_stencil);
		if (manual != current_depth_stencil_resources.end() && manual->second.identity == data.override_identity)
		{
			eligible_ids.push_back(manual->second.identity);
			observe_candidate(data, manual->second, frame_width, frame_height);
			const bool rendered = manual->second.last_used_in_frame == device_data->frame_index &&
				manual->second.last_frame_stats.total.has_work();
			data.manual_recovery_probing = data.manual_recovery.update(automatic ? data.override_identity : 0,
				rendered, device_data->frame_index, GetTickCount64());
		}
		else
		{
			reshade::log::message(reshade::log::level::info, "Sunshine 3D depth: manual buffer was destroyed or recreated; returning to automatic selection.");
			data.override_depth_stencil = { 0 };
			data.override_identity = 0;
		}
	}
	if (data.override_depth_stencil == 0)
		data.manual_recovery = {};
	if (!automatic && (data.native_depth_requested || data.raw_scene_requested) && data.override_depth_stencil == 0 && selected_depth_stencil_info != nullptr)
	{
		// Disabling content-based selection keeps the original statistics choice,
		// but its native calibration still needs current capture-region metadata.
		// Observe only that choice; never admit a challenger or overrule a manual row.
		eligible_ids.push_back(selected_depth_stencil_info->identity);
		observe_candidate(data, *selected_depth_stencil_info, frame_width, frame_height);
	}
	// Every eligible ID came from this live snapshot; a live manual row is
	// included even while inactive. This one pass therefore removes destroyed
	// and filtered candidates before consuming readbacks or choosing a source.
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
		choice = data.selection.choose(device_data->frame_index, current_id);
		if (data.manual_recovery_probing && choice.id != data.override_identity &&
			data.selection.confirmed_since(choice.id, device_data->frame_index, data.manual_recovery.inactive_frame()))
		{
			char message[256];
			std::snprintf(message, std::size(message),
				"Sunshine 3D depth: unused manual source=%llu replaced by freshly confirmed scene source=%llu; returning to automatic selection.",
				static_cast<unsigned long long>(data.override_identity), static_cast<unsigned long long>(choice.id));
			reshade::log::message(reshade::log::level::info, message);
			data.override_depth_stencil = { 0 };
			data.override_identity = 0;
			data.manual_recovery = {};
			data.manual_recovery_probing = false;
		}
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
		const bool comparison_pending = automatic && data.raw_scene_requested && data.override_depth_stencil == 0 &&
			choice.id == current_id && data.selection.pending_challenger_id() == data.challenger_id;
		if (completed_sample->token == data.challenger_id &&
			(++data.challenger_samples < 3 || comparison_pending))
			data.probe_target_id = data.challenger_id;
		else if (!sunshine_depth_probe::keep_unrelated_probe(data.raw_scene_requested && completed_other_raw_sample,
			completed_sample->token, data.probe_target_id))
			release_challenger(runtime, data);
	}

	if (data.override_depth_stencil != 0)
	{
		const auto it = find_depth_selection(current_depth_stencil_resources, data.override_depth_stencil);
		if (it != current_depth_stencil_resources.end() && it->second.identity == data.override_identity)
		{
			selected_depth_stencil = it->first;
			selected_depth_stencil_info = &it->second;
			selection_reason = "manual override";
		}
		else
		{
			data.override_depth_stencil = { 0 };
			data.override_identity = 0;
		}
	}

	const bool manual = data.override_depth_stencil != 0;
	data.selection_reason = selection_reason;

	if (!automatic || (manual && !data.manual_recovery_probing) || (data.probe_target_id != 0 &&
		std::find(eligible_ids.begin(), eligible_ids.end(), data.probe_target_id) == eligible_ids.end()))
		release_challenger(runtime, data);

	const resource_view prev_shader_resource = data.selected_shader_resource;
	const bool previous_capture_ready = data.capture_ready;
	auto *member_binding = select_captured_depth(runtime, cmd_list, data, *device_data, current_depth_stencil_resources,
		automatic, manual, frame_width, frame_height, selected_depth_stencil, selected_depth_stencil_info);
	const uint64_t selected_id = selected_depth_stencil_info ? selected_depth_stencil_info->identity : 0;
	if (member_binding)
	{
		if (data.selected_depth_stencil != 0 && !owns_member_view(data, prev_shader_resource))
			device_data->untrack_depth_stencil(device, data.selected_depth_stencil);
		data.selected_depth_stencil = selected_depth_stencil;
		data.selected_identity = selected_id;
		data.selected_desc = selected_depth_stencil_info->desc;
		data.selected_shader_resource = member_binding->view;
		data.using_backup_texture = true;
	}
	data.capture_ready = false;
	data.selected_record = {};
	data.selected_capture = {};
	data.binding_state = current_depth_stencil_resources.empty() ? depth_binding_state::waiting_game :
		(any_filtered && any_recent_activity ? depth_binding_state::filtered : depth_binding_state::waiting_activity);
	data.binding_detail = current_depth_stencil_resources.empty() ? "Waiting for the game to create a depth buffer." :
		(any_filtered && any_recent_activity ? "Depth buffers exist, but none meets the current format, aspect or sample filters." :
		(any_warming ? "New depth buffers are warming up; waiting for their next rendered frames." : "Depth buffers exist; waiting for active game rendering."));
	if (data.capture_wait_detail)
	{
		data.binding_state = depth_binding_state::waiting_capture;
		data.binding_detail = data.capture_wait_detail;
	}

	if (selected_depth_stencil != 0) do
	{
		assert(selected_depth_stencil_info != nullptr);

		const device_api api = device->get_api();

		depth_stencil_backup *depth_stencil_backup = device_data->find_depth_stencil_backup(selected_depth_stencil);

		if (selected_depth_stencil != data.selected_depth_stencil || selected_id != data.selected_identity || data.selected_shader_resource == 0 ||
			((s_preserve_depth_buffers || data.raw_scene_requested) && depth_stencil_backup == nullptr))
		{
			// Untrack previous depth-stencil first, so that backup texture can potentially be reused
			if (data.selected_depth_stencil != 0)
			{
				if (!owns_member_view(data, data.selected_shader_resource))
					device_data->untrack_depth_stencil(device, data.selected_depth_stencil);

				data.using_backup_texture = false;
				data.selected_depth_stencil = { 0 };
			}
			data.selected_shader_resource = { 0 };
			data.selected_identity = 0;
			data.selected_desc = {};

			// Create two-dimensional resource view to the first level and layer of the depth-stencil resource
			resource_view_desc srv_desc(api != device_api::opengl && api != device_api::vulkan ? format_to_default_typed(selected_depth_stencil_info->desc.texture.format) : selected_depth_stencil_info->desc.texture.format);

			// Need to create backup texture only if doing backup copies or original resource does not support shader access (which is necessary for binding it to effects)
			// Also always create a backup texture in D3D12 or Vulkan to circument problems in case application makes use of resource aliasing
			if (s_preserve_depth_buffers || automatic || data.raw_scene_requested || (selected_depth_stencil_info->desc.usage & resource_usage::shader_resource) == 0 || selected_depth_stencil_info->desc.texture.samples > 1 || (api == device_api::d3d12 || api == device_api::vulkan))
			{
				depth_stencil_backup = device_data->track_depth_stencil_for_backup(device, selected_depth_stencil, selected_depth_stencil_info->desc);

				// Abort in case backup texture creation failed
				if (depth_stencil_backup == nullptr)
				{
					data.binding_state = depth_binding_state::binding_failed;
					data.binding_detail = "The chosen buffer could not create a depth backup; no depth is currently bound.";
					break;
				}
				assert(depth_stencil_backup->backup_texture != 0);

				depth_stencil_backup->frame_width = frame_width;
				depth_stencil_backup->frame_height = frame_height;

				if (s_preserve_depth_buffers)
					reshade::get_config_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", depth_stencil_backup->force_clear_index);
				else
					depth_stencil_backup->force_clear_index = 0;

				// Avoid recreating shader resource view when the backup texture did not change
				if (prev_shader_resource == 0 || device->get_resource_from_view(prev_shader_resource) != depth_stencil_backup->backup_texture)
				{
					if (api == device_api::d3d9)
						srv_desc.format = format::r32_float; // Same format as backup texture, as set in 'track_depth_stencil_for_backup'

					if (!device->create_resource_view(depth_stencil_backup->backup_texture, resource_usage::shader_resource, srv_desc, &data.selected_shader_resource))
					{
						device_data->untrack_depth_stencil(device, selected_depth_stencil);
						data.binding_state = depth_binding_state::binding_failed;
						data.binding_detail = "The chosen buffer could not create a shader view; no depth is currently bound.";
						break;
					}
				}
				else
					data.selected_shader_resource = prev_shader_resource;

				data.using_backup_texture = true;
			}
			else
			{
				if (!device->create_resource_view(selected_depth_stencil, resource_usage::shader_resource, srv_desc, &data.selected_shader_resource))
				{
					data.binding_state = depth_binding_state::binding_failed;
					data.binding_detail = "The chosen buffer cannot be sampled; no depth is currently bound.";
					break;
				}

				assert(!data.using_backup_texture);
			}

			data.selected_depth_stencil = selected_depth_stencil;
			data.selected_identity = selected_id;
			data.selected_desc = selected_depth_stencil_info->desc;
		}
		// Clears alone cannot make a manual source ready; raster, mesh and copied depth can.
		const bool source_active = selected_depth_stencil_info->last_used_in_frame == device_data->frame_index &&
			selected_depth_stencil_info->last_frame_stats.total.has_work();

		if (data.using_backup_texture)
		{
			assert(depth_stencil_backup != nullptr && depth_stencil_backup->backup_texture != 0);
			const resource backup_texture = depth_stencil_backup->backup_texture;

			preserve_depth_at_effects(runtime, cmd_list, *device_data, selected_depth_stencil, selected_id,
				member_binding ? member_binding->capture_after : 0);

			cmd_list->barrier(backup_texture, resource_usage::copy_dest, resource_usage::shader_resource);
		}
		else
		{
			// Unset current depth-stencil view, in case it is bound to an effect as a shader resource (which will fail if it is still bound on output)
			if (api <= device_api::d3d11)
				cmd_list->bind_render_targets_and_depth_stencil(0, nullptr);

			cmd_list->barrier(selected_depth_stencil, resource_usage::depth_stencil | resource_usage::shader_resource, resource_usage::shader_resource);
		}
		const auto captured = capture_record(runtime, data, *device_data, data.selected_depth_stencil,
			data.selected_identity, data.selected_shader_resource, data.using_backup_texture,
			member_binding ? member_binding->capture_after : 0);
		data.selected_record = captured.record;
		data.selected_capture = captured.metadata;
		data.capture_ready = captured.record.renderable(device_data->native_present_index, device_data->frame_index, data.runtime_epoch);
		for (unsigned i = 0; i != data.captures.count; ++i)
			if (data.captures.records[i].identity == data.selected_identity)
			{
				data.captures.records[i] = captured.record;
			}
		data.binding_state = data.capture_ready ? depth_binding_state::ready : depth_binding_state::waiting_capture;
		data.binding_detail = data.capture_ready ? "Depth is bound and its current frame is available." :
			(source_active ? "Current buffer is selected; waiting for a valid copy before clear or unbind." : "Current buffer is selected but inactive; waiting for it to render again.");
	} while (0);
	else do
	{
		// Untrack any existing depth-stencil selected in previous frames
		if (data.selected_depth_stencil != 0)
		{
			if (!owns_member_view(data, data.selected_shader_resource))
				device_data->untrack_depth_stencil(device, data.selected_depth_stencil);

			data.using_backup_texture = false;
			data.selected_depth_stencil = { 0 };
			data.selected_shader_resource = { 0 };
		}
		data.selected_identity = 0;
		data.selected_desc = {};
	} while (0);

	if (prev_shader_resource != data.selected_shader_resource || previous_capture_ready != data.capture_ready)
	{
		update_effect_runtime(runtime);
		if (prev_shader_resource != data.selected_shader_resource && !owns_member_view(data, prev_shader_resource))
			device->destroy_resource_view(prev_shader_resource);
	}
	const bool selection_changed = data.last_logged_id != data.anchor_identity || data.last_logged_manual != manual;
	const bool binding_changed = data.last_logged_state != data.binding_state;
	// The game may alternate its tagged allocation every present. Ordinary
	// source changes share the status rate limit; manual handovers/errors remain
	// immediate. Logging must never turn physical rotation into per-frame I/O.
	if (data.status_log.due(GetTickCount64(), selection_changed || binding_changed,
		data.last_logged_manual != manual || (binding_changed && data.binding_state == depth_binding_state::binding_failed)))
	{
		char message[512];
		std::snprintf(message, std::size(message),
			"Sunshine 3D depth: %s; source=%llu %ux%u %s; %s; %s; %s",
			binding_state_name(data.binding_state), static_cast<unsigned long long>(data.selected_identity),
			data.selected_desc.texture.width, data.selected_desc.texture.height, format_to_string(data.selected_desc.texture.format), manual ? "manual" : "auto",
			data.selection_reason, data.binding_detail);
		reshade::log::message(data.binding_state == depth_binding_state::binding_failed ? reshade::log::level::error : reshade::log::level::info, message);
		data.last_logged_id = data.anchor_identity;
		data.last_logged_manual = manual;
		data.last_logged_state = data.binding_state;
	}
	prepare_probe_target(runtime, cmd_list, data, *device_data, frame_width, frame_height);
	data.native_access_present = native_present_index;
	data.native_access_open = true;
	if (s_frame_activity_trace)
		trace_activity_effect(runtime, data, *device_data, false, data.binding_detail);
}
static void set_generic_sampling_state(effect_runtime *runtime, command_list *commands, const generic_depth_data &data, bool reading)
{
	if (data.selected_shader_resource == 0) return;
	const resource texture = data.using_backup_texture ?
		runtime->get_device()->get_resource_from_view(data.selected_shader_resource) : data.selected_depth_stencil;
	const resource_usage resting = data.using_backup_texture ? resource_usage::copy_dest : resource_usage::depth_stencil | resource_usage::shader_resource;
	if (reading && runtime->get_device()->get_api() <= device_api::d3d11)
		commands->bind_render_targets_and_depth_stencil(0, nullptr);
	commands->barrier(texture, reading ? resting : resource_usage::shader_resource, reading ? resource_usage::shader_resource : resting);
}
static void finish_depth_frame(effect_runtime *runtime, command_list *cmd_list)
{
	auto &data = *runtime->get_private_data<generic_depth_data>();
	data.native_access_open = false;
	if (sunshine_streamline::provider::finish(runtime, cmd_list)) return;

	set_generic_sampling_state(runtime, cmd_list, data, false);
	sample_probe_target(runtime, cmd_list, data);
}

void sunshine_depth::set_native_driver(effect_runtime *runtime, bool enabled)
{
	if (auto *data = runtime->get_private_data<generic_depth_data>()) data->native_driver = enabled;
}
bool sunshine_depth::begin_native_frame(effect_runtime *runtime, command_list *commands)
{
	auto *data = runtime->get_private_data<generic_depth_data>();
	if (!data || !data->native_driver) return false;
	begin_depth_frame(runtime, commands);
	return true;
}
void sunshine_depth::end_native_frame(effect_runtime *runtime, command_list *commands)
{
	if (runtime->get_private_data<generic_depth_data>()) finish_depth_frame(runtime, commands);
}
static void on_begin_render_effects(effect_runtime *runtime, command_list *commands, resource_view, resource_view)
{
	auto *data = runtime->get_private_data<generic_depth_data>();
	if (!data) return;
	if (!data->native_driver) { begin_depth_frame(runtime, commands); return; }
	// Other user effects may still consume DEPTH. Reopen only its read state;
	// selection, capture and numeric sampling already ran once at native present.
	data->native_effects_lease = !sunshine_streamline::provider::selected(runtime) && data->selected_shader_resource != 0;
	if (data->native_effects_lease) set_generic_sampling_state(runtime, commands, *data, true);
}
static void on_finish_render_effects(effect_runtime *runtime, command_list *commands, resource_view, resource_view)
{
	auto *data = runtime->get_private_data<generic_depth_data>();
	if (!data) return;
	if (!data->native_driver) { finish_depth_frame(runtime, commands); return; }
	if (data->native_effects_lease) set_generic_sampling_state(runtime, commands, *data, false);
	data->native_effects_lease = false;
}

static const char *const format_to_string(format format)
{
	switch (format)
	{
	case format::d16_unorm:
	case format::r16_typeless:
		return "D16  ";
	case format::d16_unorm_s8_uint:
		return "D16S8";
	case format::d24_unorm_x8_uint:
		return "D24X8";
	case format::d24_unorm_s8_uint:
	case format::r24_g8_typeless:
		return "D24S8";
	case format::d32_float:
	case format::r32_float:
	case format::r32_typeless:
		return "D32  ";
	case format::d32_float_s8_uint:
	case format::r32_g8_typeless:
		return "D32S8";
	case format::intz:
		return "INTZ ";
	default:
		return "     ";
	}
}

static void set_manual_depth_override(generic_depth_data &data, resource depth_stencil, uint64_t identity)
{
	data.override_depth_stencil = depth_stencil;
	data.override_identity = identity;
	data.manual_recovery = {};
	data.manual_recovery_probing = false;
	// Changing the override does not invalidate captured contents or their history.
	// The next frame applies ordinary lifetime/eligibility checks and changes the
	// binding only if the chosen source changes. In particular, unpinning a useful
	// current source must not destroy its backup or restart selection from workload.
}

static void reset_depth_selection(effect_runtime *runtime, generic_depth_data &data)
{
	device *const device = runtime->get_device();
	auto *device_data = device->get_private_data<generic_depth_device_data>();
	data.manual_recovery = {};
	data.manual_recovery_probing = false;
	release_challenger(runtime, data);
	data.selection = sunshine_depth::selection_policy {};
	data.discovery_hint = {};
	data.observed_ids.clear();
	// Reset selected depth-stencil to force re-creation of resources next frame (like the backup texture)
	if (data.selected_shader_resource != 0 || std::any_of(data.member_views.begin(), data.member_views.end(),
		[](const auto &entry) { return entry.view != 0; }))
	{
		command_queue *const queue = runtime->get_command_queue();

		queue->wait_idle(); // Ensure resource view is no longer in-use before destroying it
		if (data.selected_shader_resource != 0 && !owns_member_view(data, data.selected_shader_resource))
		{
			device->destroy_resource_view(data.selected_shader_resource);
			device_data->untrack_depth_stencil(device, data.selected_depth_stencil);
		}
	}
	retire_member_views(runtime, data, *device_data, true);
	data.capture_budget.reset();
	data.anchor_depth_stencil = {}; data.anchor_identity = 0;
	data.raw_roster = {}; data.raw_members = {};
	data.captures = {}; data.selected_record = {}; data.selected_capture = {};

	data.using_backup_texture = false;
	data.selected_depth_stencil = { 0 };
	data.selected_shader_resource = { 0 };
	data.selected_identity = 0;
	data.selected_desc = {};
	data.capture_ready = false;
	data.binding_state = depth_binding_state::waiting_capture;
	data.binding_detail = "Applying depth selection settings; waiting for the next frame.";
	data.selection = sunshine_depth::selection_policy {};
	data.observed_ids.clear();
	data.observed_regions.clear();
	data.native_calibrations.clear();
	data.native_access_open = false;
	data.native_ui_state = native_depth_ui_state::waiting_depth;
	data.native_ui_samples = 0;

	update_effect_runtime(runtime);
}

static sunshine_streamline::provider::source_status describe_streamline_source(effect_runtime *runtime)
{
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	uint64_t present;
	{
		const std::shared_lock<std::shared_mutex> status_lock(s_mutex);
		present = device_data ? device_data->native_present_index : std::numeric_limits<uint64_t>::max();
	}
	return sunshine_streamline::provider::describe(runtime, present);
}

// Capture policy remains unchanged; opening these controls never changes it.
static void draw_capture_settings(device *device, bool &force_reset)
{
	if (ImGui::Checkbox("Automatically select scene depth", &s_sunshine_auto))
	{
		reshade::set_config_value(nullptr, "SUNSHINE_DEPTH", "AutoSelectSceneDepth", s_sunshine_auto);
		force_reset = true;
	}
	ImGui::SetItemTooltip("Uses confirmed game depth from Streamline or NGX. Other games use generic depth detection. A missing provider frame stays 2D until current depth returns. Reduced rendering resolutions are supported. A checked buffer takes manual priority while it renders.");
	ImGui::Spacing();

	const char *const draw_stats_heuristic_items[] = {
		"Default",
		"Higher vertices",
		"Higher draw calls"
	};
	int draw_heuristic = static_cast<int>(s_draw_stats_heuristic);
	if (ImGui::Combo("Draw stats heuristic", &draw_heuristic, draw_stats_heuristic_items, static_cast<int>(std::size(draw_stats_heuristic_items))))
	{
		s_draw_stats_heuristic = static_cast<draw_stats_heuristic>(draw_heuristic);
		reshade::set_config_value(nullptr, "DEPTH", "DrawStatsHeuristic", static_cast<unsigned int>(s_draw_stats_heuristic));
		force_reset = true;
	}

	const char *const aspect_ratio_heuristic_items[] = {
		"None",
		"Similar aspect ratio",
		"Multiples of resolution (for DLSS or resolution scaling)",
		"Match resolution exactly",
		"Match custom width and height exactly"
	};
	int aspect_heuristic = static_cast<int>(s_aspect_ratio_heuristic);
	if (ImGui::Combo("Aspect ratio heuristic", &aspect_heuristic, aspect_ratio_heuristic_items, static_cast<int>(std::size(aspect_ratio_heuristic_items))))
	{
		s_aspect_ratio_heuristic = static_cast<aspect_ratio_heuristic>(aspect_heuristic);
		reshade::set_config_value(nullptr, "DEPTH", "UseAspectRatioHeuristics", static_cast<unsigned int>(s_aspect_ratio_heuristic));
		force_reset = true;
	}

	if (s_aspect_ratio_heuristic == aspect_ratio_heuristic::match_custom_resolution_exactly)
	{
		if (ImGui::InputInt2("Filter by width and height", reinterpret_cast<int *>(s_custom_resolution_filtering)))
		{
			reshade::set_config_value(nullptr, "DEPTH", "FilterResolutionWidth", s_custom_resolution_filtering[0]);
			reshade::set_config_value(nullptr, "DEPTH", "FilterResolutionHeight", s_custom_resolution_filtering[1]);
			force_reset = true;
		}
	}

	const char *const depth_format_items[] = { // Needs to match switch in 'check_depth_format' above
		"All",
		"D16  ",
		"D16S8",
		"D24X8",
		"D24S8",
		"D32  ",
		"D32S8",
		"INTZ "
	};
	if (ImGui::Combo("Filter by depth buffer format", reinterpret_cast<int *>(&s_format_filtering), depth_format_items, static_cast<int>(std::size(depth_format_items))))
	{
		reshade::set_config_value(nullptr, "DEPTH", "FilterFormat", s_format_filtering);
		force_reset = true;
	}

	if (bool copy_before_clear_operations = s_preserve_depth_buffers != 0;
		ImGui::Checkbox("Copy depth buffer before clear operations", &copy_before_clear_operations))
	{
		s_preserve_depth_buffers = copy_before_clear_operations ? 1 : 0;
		reshade::set_config_value(nullptr, "DEPTH", "DepthCopyBeforeClears", s_preserve_depth_buffers);
		force_reset = true;
	}

	if (s_preserve_depth_buffers == 0)
		ImGui::SetItemTooltip("Enable this when the depth buffer is empty");

	const bool is_d3d12_or_vulkan = device->get_api() == device_api::d3d12 || device->get_api() == device_api::vulkan;

	if (s_preserve_depth_buffers || is_d3d12_or_vulkan)
	{
		if (bool copy_before_fullscreen_draws = s_preserve_depth_buffers == 2;
			ImGui::Checkbox(is_d3d12_or_vulkan ? "Copy depth buffer during frame to prevent artifacts" : "Copy depth buffer before fullscreen draw calls", &copy_before_fullscreen_draws))
		{
			s_preserve_depth_buffers = copy_before_fullscreen_draws ? 2 : 1;
			reshade::set_config_value(nullptr, "DEPTH", "DepthCopyBeforeClears", s_preserve_depth_buffers);
		}
	}
}

static void draw_settings_overlay(effect_runtime *runtime)
{
	if (!sunshine_game3d::draw(runtime))
		return;
	if (!s_registered)
	{
		ImGui::TextWrapped("%s", s_status_message);
		return;
	}
	if (runtime == nullptr || runtime->get_private_data<generic_depth_data>() == nullptr)
	{
		ImGui::TextWrapped("Waiting for the game to create a depth selection runtime.");
		return;
	}
	bool force_reset = false;
	auto &data = *runtime->get_private_data<generic_depth_data>();
	device *const device = runtime->get_device();
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();
	const bool manual = data.override_depth_stencil != 0;
	const auto provided = describe_streamline_source(runtime);
	const bool game_provided = provided.selected && !manual;
	const char *provider_name = provided.provider == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "Streamline";
	bool generic_current = false;
	{
		const std::shared_lock<std::shared_mutex> status_lock(s_mutex);
		generic_current = !provided.selected && device_data && data.capture_ready && data.selected_shader_resource != 0 &&
			data.selected_depth_stencil != 0 && data.native_access_present == device_data->native_present_index;
	}
	if (game_provided)
	{
		// Ownership survives capture gaps. The dimensions describe the last valid
		// source, not a claim that the current presentation has usable depth.
		if (provided.last_valid.resource)
			ImGui::Text("Depth: %s | %ux%u | game provided", provider_name,
				provided.last_valid.width, provided.last_valid.height);
		else
			ImGui::Text("Depth: %s | waiting for first depth", provider_name);
		ImGui::SetItemTooltip("The game-provided source remains selected through capture gaps. Dimensions are from its last valid capture. Current availability and depth freshness are shown in the status rows.");
	}
	else if (manual && (provided.selected || data.selected_depth_stencil != data.override_depth_stencil ||
		data.selected_identity != data.override_identity))
		ImGui::TextUnformatted("Depth: Generic (manual) | waiting for selected buffer");
	else if (data.selected_shader_resource != 0 && data.selected_depth_stencil != 0)
		ImGui::Text("Depth: %s | %ux%u | %s", manual ? "Generic (manual)" : "Generic",
			data.selected_desc.texture.width, data.selected_desc.texture.height,
			format_to_string(data.selected_desc.texture.format));
	else
		ImGui::Text("Depth: %s | waiting for a usable buffer", manual ? "Generic (manual)" : "Generic");
	if (!game_provided && generic_current)
	{
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
		ImGui::TextWrapped("Generic depth fallback is in use. Game-provided depth and camera data are not used.");
		ImGui::PopStyleColor();
	}
	if (data.manual_recovery_probing)
		ImGui::TextWrapped("Selected buffer is no longer rendering. Checking for replacement scene depth...");
	if (manual && ImGui::Button("Return to automatic"))
	{
		set_manual_depth_override(data, {}, 0);
		s_sunshine_auto = true;
		reshade::set_config_value(nullptr, "SUNSHINE_DEPTH", "AutoSelectSceneDepth", s_sunshine_auto);
	}
	if (manual)
		ImGui::SetItemTooltip("Let Sunshine choose the scene-depth buffer again.");

	if (!provided.selected && data.binding_state != depth_binding_state::ready)
		ImGui::TextWrapped("%s", data.binding_detail);
	if (game_provided)
	{
		// Report freshness over a completed window instead of alternating
		// Ready/Previous on every frame-generation presentation.
		const auto &stats = provided.presentations;
		if (const auto total = stats.total())
		{
			const double percent = 100.0 / static_cast<double>(total);
			ImGui::Text("Fresh: %.1f%% | Previous: %.1f%% | Unavailable: %.1f%%",
				stats.fresh * percent, stats.reused * percent, stats.unavailable * percent);
			ImGui::TextDisabled("Last %.0f s | %llu presentations", stats.window_ms / 1000.0,
				static_cast<unsigned long long>(total));
		}
		else
			ImGui::TextDisabled("Depth freshness: collecting samples (5-second window)");
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("Updates once per second. Fresh = newly copied depth; Previous = reused depth; Unavailable = no usable depth.\n"
				"All observed presentations are counted. This does not identify rendered or generated color frames.");
	}
	if (!ImGui::CollapsingHeader("Depth & troubleshooting"))
		return;
	ImGui::Indent();
	if (!sunshine_game3d::draw_diagnostics(runtime))
	{
		ImGui::Unindent();
		return;
	}
	bool native_effect_enabled = false;
	if (data.native_depth_requested && runtime->get_effects_state())
	{
		bool game_effect_enabled = false;
		runtime->enumerate_techniques("SunshineGame3D.fx", [&game_effect_enabled](effect_runtime *owner, effect_technique technique) {
			char name[128] {};
			owner->get_technique_name(technique, name);
			if (std::strcmp(name, "SunshineGame3D") == 0 && owner->get_technique_state(technique))
				game_effect_enabled = true;
		});
		// Match the exporter's primary-effect priority even if the user toggled
		// techniques after this frame's capture-identity observation.
		if (!game_effect_enabled)
			runtime->enumerate_techniques("SunshineDepth3D.fx", [&native_effect_enabled](effect_runtime *owner, effect_technique technique) {
				char name[128] {};
				owner->get_technique_name(technique, name);
				if (std::strcmp(name, "SunshineDepth3D") == 0 && owner->get_technique_state(technique))
					native_effect_enabled = true;
			});
	}
	if (native_effect_enabled)
	{
		native_depth_ui_state status = data.native_ui_state;
		{
			const std::shared_lock<std::shared_mutex> status_lock(s_mutex);
			if (device_data == nullptr || data.native_ui_present != device_data->native_present_index)
				status = native_depth_ui_state::waiting_depth;
		}
		switch (status)
		{
		case native_depth_ui_state::calibrating:
			ImGui::Text("Game 3D: calibrating depth (%u/3 samples).", data.native_ui_samples);
			break;
		case native_depth_ui_state::waiting_orientation:
			ImGui::TextWrapped("Game 3D: depth direction is unknown. In Sunshine 3D's shader controls, set Depth direction to Normal depth or Reversed depth.");
			break;
		case native_depth_ui_state::ready:
			ImGui::TextUnformatted("Game 3D: depth is calibrated and ready.");
			break;
		default:
			ImGui::TextUnformatted("Game 3D: waiting for current depth.");
			break;
		}
	}
	// Request optional candidate statistics only while this list is visible.
	// Source routing and the shared capture lifetime stay independent of the UI.
	if (ImGui::CollapsingHeader("Manual depth selection"))
	{
		ImGui::Indent();
		const bool is_d3d12_or_vulkan = device->get_api() == device_api::d3d12 || device->get_api() == device_api::vulkan;
		request_depth_candidate_list(device_data);
		if (game_provided)
		{
			ImGui::TextWrapped("The game supplies depth automatically. Check a buffer below only to override it manually.");
		}
		else if (data.binding_state == depth_binding_state::ready)
			ImGui::TextWrapped("%s", data.binding_detail);
		ImGui::TextWrapped("ACTIVE marks the buffer in use. Check a row only to select it manually; automatic selection leaves the boxes unchecked.");
		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Spacing();

		std::shared_lock<std::shared_mutex> lock(s_mutex);

		const bool no_buffers = device_data == nullptr || device_data->depth_stencil_resources.empty();
		const uint64_t list_frame = device_data ? device_data->frame_index : 0;
		std::vector<std::pair<resource, depth_stencil_resource>> sorted_item_list;
		if (!no_buffers)
			sorted_item_list.assign(device_data->depth_stencil_resources.begin(), device_data->depth_stencil_resources.end());
		// Unlock while calling into device below
		lock.unlock();
		resource active_depth_stencil = {};
		const uint64_t active_identity = game_provided && provided.ready ? provided.current.identity :
			generic_current ? data.selected_identity : 0;
		if (game_provided && provided.ready && active_identity)
		{
			// Match the native lifetime, not dimensions or a reusable address.
			// Prefer the exact interface row if the inventory contains aliases.
			for (const auto &[depth_stencil, info] : sorted_item_list)
			{
				if (!info.api_identity || info.identity != active_identity) continue;
				if (active_depth_stencil == 0 || depth_stencil.handle == provided.current.resource)
					active_depth_stencil = depth_stencil;
			}
		}
		else if (generic_current)
			active_depth_stencil = data.selected_depth_stencil;

		if (game_provided && provided.ready && active_depth_stencil == 0)
		{
			// Compute-generated/API-only depth may never appear as a ReShade DSV.
			// Show its real source without inventing a selectable Generic buffer.
			ImGui::TextColored(ImGui::GetStyle().Colors[ImGuiCol_ButtonActive],
				"%ux%u | %s | ACTIVE (%s)", provided.current.width, provided.current.height,
				format_to_string(static_cast<format>(provided.current.format)), provider_name);
			ImGui::SetItemTooltip("Game-provided depth: 0x%016llx\nThis source is not a selectable generic buffer.",
				static_cast<unsigned long long>(provided.current.resource));
		}
		if (no_buffers)
			ImGui::TextUnformatted(game_provided ? "No generic buffers available for manual selection." : "No depth buffers found.");

		// Sort pointer list so that added/removed items do not change the GUI much
		std::sort(sorted_item_list.begin(), sorted_item_list.end(),
			[active_depth_stencil](const std::pair<resource, depth_stencil_resource> &a, const std::pair<resource, depth_stencil_resource> &b) {
				if ((a.first == active_depth_stencil) != (b.first == active_depth_stencil))
					return a.first == active_depth_stencil;
				return ((a.second.desc.texture.width > b.second.desc.texture.width || (a.second.desc.texture.width == b.second.desc.texture.width && a.second.desc.texture.height > b.second.desc.texture.height)) ||
						(a.second.desc.texture.width == b.second.desc.texture.width && a.second.desc.texture.height == b.second.desc.texture.height && a.first < b.first));
			});

		uint32_t frame_width, frame_height;
		runtime->get_screenshot_width_and_height(&frame_width, &frame_height);

		bool has_msaa_depth_stencil = false;
		bool has_no_clear_operations = false;
		unsigned visible_rows = 0;

		for (const auto &[depth_stencil, info] : sorted_item_list)
		{
			const bool selected = active_identity && depth_stencil == active_depth_stencil && info.identity == active_identity;
			if (!selected && (info.last_used_in_frame == std::numeric_limits<uint64_t>::max() || list_frame > (info.last_used_in_frame + 30)))
				continue; // Hide from list when not used for a couple of frames
			++visible_rows;

			const bool inactive = info.last_used_in_frame == std::numeric_limits<uint64_t>::max() || list_frame > (info.last_used_in_frame + 5);
			bool disabled = inactive;
			if (info.desc.texture.samples > 1 && !device->check_capability(device_caps::resolve_depth_stencil)) // Disable widget for multisampled textures
				has_msaa_depth_stencil = disabled = true;

			const bool candidate =
				!(!info.last_frame_stats.total.has_work() || (!s_sunshine_auto && info.last_frame_stats.total.vertices <= 3 && info.last_frame_stats.total.drawcalls_indirect == 0 && info.last_frame_stats.total.scene_writes == 0)) &&
				!(s_format_filtering != 0 && !check_depth_format(info.desc.texture.format)) &&
				candidate_matches_shape(info, frame_width, frame_height);

			char label[128];
			// Keep the checkbox ID stable when the ACTIVE label moves on rotation.
			std::snprintf(label, std::size(label), "Buffer %llu | %ux%u | %s%s###depth-%016llx-%016llx",
				static_cast<unsigned long long>(info.identity),
				info.desc.texture.width, info.desc.texture.height, format_to_string(info.desc.texture.format),
				selected ? " | ACTIVE" : "",
				static_cast<unsigned long long>(depth_stencil.handle), static_cast<unsigned long long>(info.identity));

			ImGui::BeginDisabled(disabled);
			ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyle().Colors[selected ? ImGuiCol_ButtonActive : disabled ? ImGuiCol_TextDisabled : candidate ? ImGuiCol_ButtonActive : ImGuiCol_Text]);

			if (bool value = (depth_stencil == data.override_depth_stencil && info.identity == data.override_identity);
				ImGui::Checkbox(label, &value))
			{
				set_manual_depth_override(data, value ? depth_stencil : resource { 0 }, value ? info.identity : 0);
			}

			if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			{
				const auto *evidence = data.selection.evidence(info.identity);
				ImGui::SetTooltip("Buffer: 0x%016llx | Lifetime: %llu\n"
					"Draw calls: %u (%u indirect) | Vertices: %u\n"
					"Depth quality: %s%s\n%s%s%s%s",
					static_cast<unsigned long long>(depth_stencil.handle), static_cast<unsigned long long>(info.identity),
					info.last_frame_stats.total.drawcalls, info.last_frame_stats.total.drawcalls_indirect,
					info.last_frame_stats.total.vertices,
					evidence ? content_kind_name(evidence->quality.kind) : "Not yet sampled",
					evidence && evidence->confirmed_good ? " (confirmed)" : "",
					selected && game_provided ? provider_name : "Generic",
					inactive ? " | Inactive" : "",
					info.desc.texture.samples > 1 ? " | Multisampled" : "",
					info.last_frame_stats.reversed_clear_value ? " | Reversed" : "");
			}

			ImGui::PopStyleColor();
			ImGui::EndDisabled();

			if (!provided.selected && s_preserve_depth_buffers && depth_stencil == data.selected_depth_stencil)
			{
				if (info.last_frame_stats.clears.empty())
				{
					has_no_clear_operations = !is_d3d12_or_vulkan;
					continue;
				}

				depth_stencil_backup *const depth_stencil_backup = device_data->find_depth_stencil_backup(depth_stencil);
				if (depth_stencil_backup == nullptr || depth_stencil_backup->backup_texture == 0)
					continue;

				for (uint32_t clear_index = 1; clear_index <= static_cast<uint32_t>(info.last_frame_stats.clears.size()); ++clear_index)
				{
					const clear_stats &clear_stats = info.last_frame_stats.clears[clear_index - 1];

					std::snprintf(label, std::size(label), "Clear %u%s###clear-%016llx-%016llx-%u", clear_index,
						clear_stats.copied_during_frame ? " | CAPTURED" : "",
						static_cast<unsigned long long>(depth_stencil.handle), static_cast<unsigned long long>(info.identity), clear_index);

					if (bool value = (depth_stencil_backup->force_clear_index == clear_index);
						ImGui::Checkbox(label, &value))
					{
						depth_stencil_backup->force_clear_index = value ? clear_index : 0;
						reshade::set_config_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", depth_stencil_backup->force_clear_index);
					}

					ImGui::SetItemTooltip("Draw calls: %u (%u indirect) | Vertices: %u%s",
						clear_stats.drawcalls,
						clear_stats.drawcalls_indirect,
						clear_stats.vertices,
						clear_stats.clear_op == clear_op::fullscreen_draw ? "\nFullscreen draw call" : "");
				}

				if (!is_d3d12_or_vulkan)
				{
					if (bool value = (depth_stencil_backup->force_clear_index == std::numeric_limits<uint32_t>::max());
						ImGui::Checkbox("Use last busy clear", &value))
					{
						depth_stencil_backup->force_clear_index = value ? std::numeric_limits<uint32_t>::max() : 0;
						reshade::set_config_value(nullptr, "DEPTH", "DepthCopyAtClearIndex", depth_stencil_backup->force_clear_index);
					}
					ImGui::SetItemTooltip("Choose the last clear operation with a high number of draw calls.");
				}
			}
		}

		if (visible_rows == 0 && !no_buffers)
			ImGui::TextWrapped("%s", game_provided ? "No recently active generic depth buffers; game-provided depth is shown above." :
				"No recently active depth buffers. Waiting for the game to render; check the current-buffer status above.");

		if (!game_provided && (has_msaa_depth_stencil || has_no_clear_operations))
		{
			ImGui::Spacing();
			ImGui::Separator();
			ImGui::Spacing();

			ImGui::PushTextWrapPos();
			if (has_msaa_depth_stencil)
				ImGui::TextUnformatted("Not all depth buffers are available.\nYou may have to disable MSAA (Multisample Anti-Aliasing) in the game settings for depth buffer detection to work!");
			if (has_no_clear_operations)
				ImGui::Text("No clear operations were found for the selected depth buffer.\n%s",
					s_preserve_depth_buffers != 2 ? "Try enabling \"Copy depth buffer before fullscreen draw calls\" or disable \"Copy depth buffer before clear operations\"!" : "Disable \"Copy depth buffer before clear operations\" or select a different depth buffer!");
			ImGui::PopTextWrapPos();
		}
		ImGui::Spacing();
		if (ImGui::CollapsingHeader("Advanced capture settings"))
		{
			ImGui::Indent();
			draw_capture_settings(device, force_reset);
			ImGui::Unindent();
		}
		ImGui::Unindent();
	}

	if (force_reset)
		reset_depth_selection(runtime, data);
	ImGui::Unindent();
}

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
void sunshine_game3d_test_draw_production_panel(effect_runtime *runtime, bool expanded)
{
	// Set only this isolated test window's tree state. No duplicate layout,
	// control implementation, shader values or production capture state.
	ImGui::GetStateStorage()->SetInt(ImGui::GetID("Depth & troubleshooting"), expanded);
	ImGui::GetStateStorage()->SetInt(ImGui::GetID("Manual depth selection"), expanded);
	draw_settings_overlay(runtime);
}
#endif

static void register_depth_events()
{
	s_streamline_probe_events = sunshine_streamline::enabled();
	s_streamline_source_events = sunshine_streamline::source_enabled() || sunshine_upscaler_trace::capture_enabled();
	sunshine_streamline::depth_capture::initialize(true); // Shared D3D12 capture also serves Generic fallback.
	sunshine_streamline::depth_capture::set_preservation_available(preservation_available);
	s_frame_activity_trace = false;
	reshade::get_config_value(nullptr, "SUNSHINE_DEPTH", "FrameActivityTrace", s_frame_activity_trace);
	reshade::register_event<reshade::addon_event::init_device>(sunshine_addon_lifetime::guarded<on_init_device>);
	reshade::register_event<reshade::addon_event::init_command_list>(sunshine_addon_lifetime::guarded<on_init_command_list>);
	reshade::register_event<reshade::addon_event::init_command_queue>(sunshine_addon_lifetime::guarded<on_init_command_queue>);
	reshade::register_event<reshade::addon_event::init_effect_runtime>(sunshine_addon_lifetime::guarded<on_init_effect_runtime>);
	reshade::register_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
	reshade::register_event<reshade::addon_event::destroy_command_list>(sunshine_addon_lifetime::guarded<on_destroy_command_list>);
	reshade::register_event<reshade::addon_event::destroy_command_queue>(sunshine_addon_lifetime::guarded<on_destroy_command_queue>);
	reshade::register_event<reshade::addon_event::destroy_effect_runtime>(sunshine_addon_lifetime::guarded<on_destroy_effect_runtime>);

	reshade::register_event<reshade::addon_event::create_resource>(sunshine_addon_lifetime::guarded<on_create_resource>);
	reshade::register_event<reshade::addon_event::create_resource_view>(sunshine_addon_lifetime::guarded<on_create_resource_view>);
	reshade::register_event<reshade::addon_event::init_resource>(sunshine_addon_lifetime::guarded<on_init_resource>);
	reshade::register_event<reshade::addon_event::destroy_resource>(sunshine_addon_lifetime::guarded<on_destroy_resource>);

	reshade::register_event<reshade::addon_event::draw>(sunshine_addon_lifetime::guarded<on_draw>);
	reshade::register_event<reshade::addon_event::draw_indexed>(sunshine_addon_lifetime::guarded<on_draw_indexed>);
	reshade::register_event<reshade::addon_event::draw_or_dispatch_indirect>(sunshine_addon_lifetime::guarded<on_draw_indirect>);
	reshade::register_event<reshade::addon_event::dispatch_mesh>(sunshine_addon_lifetime::guarded<on_dispatch_mesh>);
	reshade::register_event<reshade::addon_event::copy_resource>(sunshine_addon_lifetime::guarded<on_copy_resource>);
	reshade::register_event<reshade::addon_event::copy_texture_region>(sunshine_addon_lifetime::guarded<on_copy_texture_region>);
	reshade::register_event<reshade::addon_event::copy_buffer_to_texture>(sunshine_addon_lifetime::guarded<on_copy_buffer_to_texture>);
	reshade::register_event<reshade::addon_event::resolve_texture_region>(sunshine_addon_lifetime::guarded<on_resolve_texture_region>);
	reshade::register_event<reshade::addon_event::barrier>(sunshine_addon_lifetime::guarded<on_content_barrier>);
	reshade::register_event<reshade::addon_event::end_render_pass>(sunshine_addon_lifetime::guarded<on_end_render_pass>);
	if (s_streamline_probe_events)
	{
		reshade::register_event<reshade::addon_event::close_command_list>(sunshine_addon_lifetime::guarded<on_close>);
	}
	reshade::register_event<reshade::addon_event::bind_viewports>(sunshine_addon_lifetime::guarded<on_bind_viewport>);
	reshade::register_event<reshade::addon_event::begin_render_pass>(sunshine_addon_lifetime::guarded<on_begin_render_pass_with_depth_stencil>);
	reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(sunshine_addon_lifetime::guarded<on_bind_depth_stencil>);
	reshade::register_event<reshade::addon_event::clear_depth_stencil_view>(sunshine_addon_lifetime::guarded<on_clear_depth_stencil>);

	reshade::register_event<reshade::addon_event::reset_command_list>(sunshine_addon_lifetime::guarded<on_reset>);
	reshade::register_event<reshade::addon_event::execute_command_list>(sunshine_addon_lifetime::guarded<on_execute_primary>);
	reshade::register_event<reshade::addon_event::execute_secondary_command_list>(sunshine_addon_lifetime::guarded<on_execute_secondary>);

	reshade::register_event<reshade::addon_event::present>(sunshine_addon_lifetime::guarded<on_present>);
	reshade::register_event<reshade::addon_event::finish_present>(sunshine_addon_lifetime::guarded<sunshine_depth::finish_present>);

	reshade::register_event<reshade::addon_event::reshade_begin_effects>(sunshine_addon_lifetime::guarded<on_begin_render_effects>);
	reshade::register_event<reshade::addon_event::reshade_finish_effects>(sunshine_addon_lifetime::guarded<on_finish_render_effects>);
	// Need to set texture binding again after reloading
	reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(sunshine_addon_lifetime::guarded<on_reload_effect_runtime>);
}
static void unregister_depth_events()
{
	reshade::unregister_event<reshade::addon_event::init_device>(sunshine_addon_lifetime::guarded<on_init_device>);
	reshade::unregister_event<reshade::addon_event::init_command_list>(sunshine_addon_lifetime::guarded<on_init_command_list>);
	reshade::unregister_event<reshade::addon_event::init_command_queue>(sunshine_addon_lifetime::guarded<on_init_command_queue>);
	reshade::unregister_event<reshade::addon_event::init_effect_runtime>(sunshine_addon_lifetime::guarded<on_init_effect_runtime>);
	reshade::unregister_event<reshade::addon_event::destroy_device>(sunshine_addon_lifetime::guarded<on_destroy_device>);
	reshade::unregister_event<reshade::addon_event::destroy_command_list>(sunshine_addon_lifetime::guarded<on_destroy_command_list>);
	reshade::unregister_event<reshade::addon_event::destroy_command_queue>(sunshine_addon_lifetime::guarded<on_destroy_command_queue>);
	reshade::unregister_event<reshade::addon_event::destroy_effect_runtime>(sunshine_addon_lifetime::guarded<on_destroy_effect_runtime>);

	reshade::unregister_event<reshade::addon_event::create_resource>(sunshine_addon_lifetime::guarded<on_create_resource>);
	reshade::unregister_event<reshade::addon_event::create_resource_view>(sunshine_addon_lifetime::guarded<on_create_resource_view>);
	reshade::unregister_event<reshade::addon_event::init_resource>(sunshine_addon_lifetime::guarded<on_init_resource>);
	reshade::unregister_event<reshade::addon_event::destroy_resource>(sunshine_addon_lifetime::guarded<on_destroy_resource>);

	reshade::unregister_event<reshade::addon_event::draw>(sunshine_addon_lifetime::guarded<on_draw>);
	reshade::unregister_event<reshade::addon_event::draw_indexed>(sunshine_addon_lifetime::guarded<on_draw_indexed>);
	reshade::unregister_event<reshade::addon_event::draw_or_dispatch_indirect>(sunshine_addon_lifetime::guarded<on_draw_indirect>);
	reshade::unregister_event<reshade::addon_event::dispatch_mesh>(sunshine_addon_lifetime::guarded<on_dispatch_mesh>);
	reshade::unregister_event<reshade::addon_event::copy_resource>(sunshine_addon_lifetime::guarded<on_copy_resource>);
	reshade::unregister_event<reshade::addon_event::copy_texture_region>(sunshine_addon_lifetime::guarded<on_copy_texture_region>);
	reshade::unregister_event<reshade::addon_event::copy_buffer_to_texture>(sunshine_addon_lifetime::guarded<on_copy_buffer_to_texture>);
	reshade::unregister_event<reshade::addon_event::resolve_texture_region>(sunshine_addon_lifetime::guarded<on_resolve_texture_region>);
	reshade::unregister_event<reshade::addon_event::barrier>(sunshine_addon_lifetime::guarded<on_content_barrier>);
	reshade::unregister_event<reshade::addon_event::end_render_pass>(sunshine_addon_lifetime::guarded<on_end_render_pass>);
	if (s_streamline_probe_events)
	{
		reshade::unregister_event<reshade::addon_event::close_command_list>(sunshine_addon_lifetime::guarded<on_close>);
	}
	reshade::unregister_event<reshade::addon_event::bind_viewports>(sunshine_addon_lifetime::guarded<on_bind_viewport>);
	reshade::unregister_event<reshade::addon_event::begin_render_pass>(sunshine_addon_lifetime::guarded<on_begin_render_pass_with_depth_stencil>);
	reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(sunshine_addon_lifetime::guarded<on_bind_depth_stencil>);
	reshade::unregister_event<reshade::addon_event::clear_depth_stencil_view>(sunshine_addon_lifetime::guarded<on_clear_depth_stencil>);

	reshade::unregister_event<reshade::addon_event::reset_command_list>(sunshine_addon_lifetime::guarded<on_reset>);
	reshade::unregister_event<reshade::addon_event::execute_command_list>(sunshine_addon_lifetime::guarded<on_execute_primary>);
	reshade::unregister_event<reshade::addon_event::execute_secondary_command_list>(sunshine_addon_lifetime::guarded<on_execute_secondary>);

	reshade::unregister_event<reshade::addon_event::present>(sunshine_addon_lifetime::guarded<on_present>);
	reshade::unregister_event<reshade::addon_event::finish_present>(sunshine_addon_lifetime::guarded<sunshine_depth::finish_present>);

	reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(sunshine_addon_lifetime::guarded<on_begin_render_effects>);
	reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(sunshine_addon_lifetime::guarded<on_finish_render_effects>);
	reshade::unregister_event<reshade::addon_event::reshade_reloaded_effects>(sunshine_addon_lifetime::guarded<on_reload_effect_runtime>);
}

static bool read_disabled_addons(std::vector<std::string> &names)
{
	size_t size = 0;
	if (!reshade::get_config_value(nullptr, "ADDON", "DisabledAddons", nullptr, &size))
		return true; // A missing or empty list is valid and has no existing entries.
	if (size == 0 || size > 65536)
		return false;
	std::vector<char> value(size, '\0');
	if (!reshade::get_config_value(nullptr, "ADDON", "DisabledAddons", value.data(), &size) || size >= value.size())
		return false;
	// The public API returns a NUL-separated array. The built-in loader matches
	// its exact plain name; an external-addon filename-qualified entry is not it.
	for (size_t offset = 0; offset < size;)
	{
		const auto end = std::find(value.begin() + offset, value.begin() + size, '\0');
		if (end == value.begin() + size)
			return false; // Do not replace a truncated or malformed existing list.
		const size_t length = static_cast<size_t>(end - value.begin()) - offset;
		names.emplace_back(value.data() + offset, length);
		offset += length + 1;
	}
	return true;
}

enum class legacy_depth_state { absent, present, unknown };

static bool readable_module_range(HMODULE module, size_t image_size, const void *address, size_t size)
{
	const auto base = reinterpret_cast<uintptr_t>(module);
	const auto begin = reinterpret_cast<uintptr_t>(address);
	if (begin < base || begin - base >= image_size || size > image_size - (begin - base))
		return false;
	for (uintptr_t cursor = begin; cursor < begin + size;)
	{
		MEMORY_BASIC_INFORMATION info = {};
		if (VirtualQuery(reinterpret_cast<const void *>(cursor), &info, sizeof(info)) != sizeof(info) ||
			info.AllocationBase != module || info.State != MEM_COMMIT || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) != 0 ||
			(info.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) == 0)
			return false;
		const auto region = reinterpret_cast<uintptr_t>(info.BaseAddress);
		if (region > cursor || info.RegionSize <= cursor - region)
			return false;
		cursor += std::min(size_t(begin + size - cursor), info.RegionSize - size_t(cursor - region));
	}
	return true;
}

static bool module_names_legacy_depth(HMODULE module, size_t image_size)
{
	// ReShade's public NAME export is a pointer variable, not a function or an
	// inline character array. Read only committed readable ranges of the held
	// module image; unrelated exports and forwarded addresses are not trusted.
	const auto exported_name = reinterpret_cast<const void *>(GetProcAddress(module, "NAME"));
	if (!readable_module_range(module, image_size, exported_name, sizeof(const char *)))
		return false;
	const char *name = nullptr;
	std::memcpy(&name, exported_name, sizeof(name));
	static constexpr char legacy_name[] = "Sunshine Depth";
	return readable_module_range(module, image_size, name, sizeof(legacy_name)) &&
		std::memcmp(name, legacy_name, sizeof(legacy_name)) == 0;
}

static legacy_depth_state legacy_depth_is_loaded(HMODULE addon_module)
{
	for (const wchar_t *name : {L"SunshineDepth.addon64", L"SunshineDepth.addon32", L"SunshineDepth.addon"})
	{
		const HMODULE legacy = GetModuleHandleW(name);
		if (legacy != nullptr && legacy != addon_module)
			return legacy_depth_state::present;
	}
	HANDLE snapshot = INVALID_HANDLE_VALUE;
	for (unsigned attempt = 0; attempt < 3; ++attempt)
	{
		snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
		if (snapshot != INVALID_HANDLE_VALUE || GetLastError() != ERROR_BAD_LENGTH)
			break;
	}
	if (snapshot == INVALID_HANDLE_VALUE)
		return legacy_depth_state::unknown;
	MODULEENTRY32W entry = {};
	entry.dwSize = sizeof(entry);
	for (BOOL more = Module32FirstW(snapshot, &entry); more; more = Module32NextW(snapshot, &entry))
	{
		if (entry.hModule == addon_module)
			continue;
		HMODULE held = nullptr;
		// Snapshot entries do not hold a loader reference. Keep the image mapped
		// while reading the NAME export if another thread unloads a module.
		if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, reinterpret_cast<LPCWSTR>(entry.modBaseAddr), &held))
			continue;
		const bool found = module_names_legacy_depth(held, entry.modBaseSize);
		FreeLibrary(held);
		if (found)
		{
			CloseHandle(snapshot);
			return legacy_depth_state::present;
		}
	}
	const DWORD enumeration_error = GetLastError();
	CloseHandle(snapshot);
	return enumeration_error == ERROR_NO_MORE_FILES ? legacy_depth_state::absent : legacy_depth_state::unknown;
}

bool sunshine_depth::initialize(HMODULE addon_module)
{
	if (s_initialized)
		return s_registered;
	s_initialized = true;
	sunshine_game3d::initialize();
	reshade::register_overlay(s_overlay_title, sunshine_addon_lifetime::guarded<draw_settings_overlay>);

	std::vector<std::string> disabled_addons;
	if (!read_disabled_addons(disabled_addons))
	{
		s_status_message = "Depth selection could not read the existing add-on settings. Add Generic Depth to [ADDON] DisabledAddons in ReShade.ini, remove the old SunshineDepth.addon64, and restart the game.";
		reshade::log::message(reshade::log::level::error, s_status_message);
		return false;
	}
	const bool builtin_disabled = std::find(disabled_addons.begin(), disabled_addons.end(), "Generic Depth") != disabled_addons.end();
	const legacy_depth_state legacy = legacy_depth_is_loaded(addon_module);
	bool changed = false;
	for (const char *name : {"Generic Depth", "Sunshine Depth"})
		if (std::find(disabled_addons.begin(), disabled_addons.end(), name) == disabled_addons.end())
		{
			disabled_addons.emplace_back(name);
			changed = true;
		}
	if (changed)
	{
		std::vector<char> value;
		for (const std::string &name : disabled_addons)
		{
			value.insert(value.end(), name.begin(), name.end());
			value.push_back('\0');
		}
		if (value.size() > 65536)
		{
			s_status_message = "Depth selection could not update the existing add-on settings. Add Generic Depth to [ADDON] DisabledAddons in ReShade.ini, remove the old SunshineDepth.addon64, and restart the game.";
			reshade::log::message(reshade::log::level::error, s_status_message);
			return false;
		}
		// Preserve every existing disabled entry. ReShade checks this current list
		// when an external add-on registers, so a legacy selector encountered later
		// in the directory scan is rejected even on this launch. Already registered
		// callbacks, including the built-in depth add-on, remain untouched.
		reshade::set_config_value(nullptr, "ADDON", "DisabledAddons", value.data(), value.size());
	}
	if (!builtin_disabled)
	{
		// ReShade 6.8 registers its built-ins before calling external AddonInit.
		// Its public API cannot remove the already registered built-in callbacks.
		s_status_message = "Depth setup has been updated. Restart the game to enable Sunshine 3D automatic depth selection. Existing ReShade depth selection remains available for this launch.";
		reshade::log::message(reshade::log::level::warning, s_status_message);
		return false;
	}
	if (legacy == legacy_depth_state::unknown)
	{
		s_status_message = "Depth selection could not verify the loaded add-ons. Restart the game after removing old Sunshine Depth add-ons; export remains available for this launch.";
		reshade::log::message(reshade::log::level::warning, s_status_message);
		return false;
	}
	if (legacy == legacy_depth_state::present)
	{
		s_status_message = "The old Sunshine Depth add-on is already loaded, possibly under a renamed file. Close the game, remove that old add-on, and restart. Keep only the Sunshine 3D add-on; automatic selection will then use its integrated depth selector.";
		reshade::log::message(reshade::log::level::warning, s_status_message);
		return false;
	}
	register_depth_events();
	s_registered = true;
	s_status_message = "Depth selection is ready.";
	reshade::log::message(reshade::log::level::info, "Sunshine 3D depth selector initialized.");
	return true;
}

void sunshine_depth::shutdown()
{
	sunshine_streamline::depth_capture::shutdown();
	sunshine_game3d::shutdown();
	if (s_registered)
		unregister_depth_events();
	if (s_initialized)
		reshade::unregister_overlay(s_overlay_title, sunshine_addon_lifetime::guarded<draw_settings_overlay>);
	s_registered = false;
	s_initialized = false;
	s_status_message = "Depth selection has not been initialized.";
}

bool sunshine_depth::active()
{
	return s_registered;
}

const char *sunshine_depth::status_message()
{
	return s_status_message;
}

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
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
		if (!ok) reshade::log::message(reshade::log::level::error, label);
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
#endif

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

bool sunshine_depth::get_latest_raw_sample(effect_runtime *runtime, sunshine_raw_scene::sample &output)
{
	output = {};
	if (!s_registered || runtime == nullptr) return false;
	const auto *data = runtime->get_private_data<generic_depth_data>();
	if (!data || !data->native_access_open || !data->capture_ready) return false;
	std::array<sunshine_raw_scene::sample, sunshine_raw_scene::maximum_sources> samples;
	const auto count = get_raw_samples(runtime, samples);
	for (std::size_t i = 0; i != count; ++i)
		if (samples[i].metadata.source.lifetime == data->selected_identity &&
			samples[i].metadata.source.native == data->selected_depth_stencil.handle)
		{ output = samples[i]; return true; }
	return false;
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

bool sunshine_depth::get_latest_camera_observation(effect_runtime *runtime,
	sunshine_camera_binding::submission &output, std::uint64_t &capture_id)
{
	output = {};
	capture_id = 0;
	if (!sunshine_streamline::enabled()) return false;
	// Reuse the production current-selection/lifetime/viewport/freshness checks.
	// This diagnostic is throttled; it does not add work to disabled gameplay.
	sunshine_raw_scene::sample sample;
	if (!get_latest_raw_sample(runtime, sample)) return false;
	auto *data = runtime->get_private_data<generic_depth_data>();
	const auto *member = raw_member(*data, sample.metadata.source.lifetime);
	if (!member) return false;
	output = member->submission;
	output.camera.proof_admitted = false;
	capture_id = sample.id;
	return true;
}

void sunshine_depth::report_camera_observations(effect_runtime *runtime, const sunshine_streamline::selected_depth &selected)
{
	if (!sunshine_streamline::enabled() || !s_registered || runtime == nullptr) return;
	auto *data = runtime->get_private_data<generic_depth_data>();
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (!data || !device_data) return;
	const auto now = GetTickCount64();
	if (now < data->next_camera_observation_report) return;
	data->next_camera_observation_report = now + 5000;
	char text[1400];
	sunshine_camera_binding::submission sample;
	std::uint64_t capture = 0;
	const bool available = get_latest_camera_observation(runtime, sample, capture);
	const auto &observation = sample.observation;
	std::snprintf(text, sizeof(text),
		"Sunshine camera sample: current_selection_sample=%d capture=%llu capture_frame=%llu source=%llu backup=%llu copy=%llu status=%s binding=%s level=%s projection_observed=%d evaluation=%llu camera_sequence=%llu frame_kind=%u numeric_frame=%llu token_generation=%llu capture_tick=%llu evaluation_tick=%llu A=%.17g B=%.17g proof_admitted=0 geometry_unchanged=1",
		available ? 1 : 0, static_cast<unsigned long long>(capture), static_cast<unsigned long long>(sample.capture_frame),
		static_cast<unsigned long long>(sample.selection.original.lifetime), static_cast<unsigned long long>(sample.selection.sampled.lifetime),
		static_cast<unsigned long long>(sample.copy.copy_id), sunshine_streamline::name(static_cast<sunshine_streamline::evidence_status>(observation.evidence_status)),
		sunshine_camera_binding::name(observation.binding), sunshine_camera_binding::name(observation.level),
		sample.projection_associated ? 1 : 0, static_cast<unsigned long long>(observation.sequence),
		static_cast<unsigned long long>(observation.camera_sequence), observation.frame_kind,
		static_cast<unsigned long long>(observation.frame_numeric), static_cast<unsigned long long>(observation.frame_generation),
		static_cast<unsigned long long>(sample.capture_ms), static_cast<unsigned long long>(observation.tick), sample.camera.A, sample.camera.B);
	reshade::log::message(reshade::log::level::info, text);

	sunshine_streamline::evaluation_snapshot evidence;
	sunshine_streamline::query_evaluation(selected, evidence);
	struct lookup {
		bool present{}, known{}, current{}, copied{}, ambiguous{}, selected{};
		std::uint64_t native{}, lifetime{}, layout{}, last_frame{};
		std::uint32_t type{}, width{}, height{}, format{};
		capture_region region;
	};
	std::array<lookup, 3> found{};
	bool map_busy = false;
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex, std::try_to_lock);
		map_busy = !lock.owns_lock();
		if (!map_busy) for (std::size_t i = 0; i != evidence.tags.size(); ++i)
		{
			const auto &tag = evidence.tags[i];
			if (!tag.present || !tag.value.native_resource) continue;
			auto &entry = found[i];
			entry.present = true; entry.native = tag.value.native_resource; entry.type = tag.value.type;
			const auto candidate = device_data->depth_stencil_resources.find(resource{entry.native});
			if (candidate == device_data->depth_stencil_resources.end()) continue;
			const auto &info = candidate->second;
			entry.known = true; entry.lifetime = info.identity; entry.layout = info.layout_epoch;
			entry.width = info.desc.texture.width; entry.height = info.desc.texture.height; entry.format = static_cast<std::uint32_t>(info.desc.texture.format);
			entry.last_frame = info.last_used_in_frame; entry.current = info.last_used_in_frame == device_data->frame_index;
			entry.copied = info.last_frame_stats.copied_during_frame; entry.ambiguous = info.last_frame_stats.copy_provenance_ambiguous;
			entry.selected = candidate->first == data->selected_depth_stencil && info.identity == data->selected_identity;
			entry.region = effective_capture_region(info);
		}
	}
	if (map_busy)
	{
		reshade::log::message(reshade::log::level::info,
			"Sunshine camera depth lookup: status=resource-map-busy lookup_available=0 geometry_unchanged=1");
		return;
	}
	for (const auto &entry : found) if (entry.present)
	{
		std::snprintf(text, sizeof(text),
			"Sunshine camera depth lookup: evaluation=%llu type=%u native=0x%llx known=%d source=%llu size=%ux%u format=%u region=%u,%u,%u,%u layout=%llu last_frame=%llu current=%d copied=%d ambiguous=%d selected=%d address_lookup_only=1 geometry_unchanged=1",
			static_cast<unsigned long long>(evidence.sequence), entry.type, static_cast<unsigned long long>(entry.native), entry.known ? 1 : 0,
			static_cast<unsigned long long>(entry.lifetime), entry.width, entry.height, entry.format,
			entry.region.x, entry.region.y, entry.region.width, entry.region.height, static_cast<unsigned long long>(entry.layout),
			static_cast<unsigned long long>(entry.last_frame), entry.current ? 1 : 0, entry.copied ? 1 : 0,
			entry.ambiguous ? 1 : 0, entry.selected ? 1 : 0);
		reshade::log::message(reshade::log::level::info, text);
	}
}

bool sunshine_depth::get_frame_depth(effect_runtime *runtime, frame_depth &output, depth_orientation orientation_override, bool request_calibration)
{
	output = {};
	if (!s_registered || runtime == nullptr)
		return false;
	if (sunshine_streamline::provider::current(runtime, output)) return output.ready;
	device *const device = runtime->get_device();
	auto *data = runtime->get_private_data<generic_depth_data>();
	auto *device_data = device->get_private_data<generic_depth_device_data>();
	// This scope outlives the inner resource-map lock; diagnostics never log
	// while that lock is held and preserve all existing return conditions.
	activity_access_trace trace { runtime, data, device_data };
	if (data != nullptr)
	{
		// This is the current renderer's request, not a latch until effect reload.
		// Passive observation releases an earlier independent-renderer request
		// without changing scene selection, histogram evidence or capture state.
		data->native_depth_requested = request_calibration;
		data->native_ui_state = request_calibration ? native_depth_ui_state::waiting_depth : native_depth_ui_state::inactive;
		data->native_ui_samples = 0;
		data->native_ui_present = request_calibration ? data->native_access_present : 0;
	}
	if (data == nullptr || device_data == nullptr) return trace.reject("missing_state");
	if (!data->native_access_open) return trace.reject("access_closed");
	if (!data->capture_ready) return trace.reject("capture_unavailable");
	if (data->selected_shader_resource == 0) return trace.reject("missing_view");

	// Binding, rendering and sampling share the immutable capture result. The
	// validation only checks that its lifetime/assignment/present is still live;
	// it never substitutes another source or newer copy into this frame.
	if (s_frame_activity_trace)
	{
		const std::shared_lock<std::shared_mutex> lock(s_mutex);
		trace.decision_present = device_data->native_present_index;
		trace.decision_frame = device_data->frame_index;
	}
	if (data->native_access_present != device_data->native_present_index ||
		!validate_capture_record(runtime, *data, *device_data, data->selected_record))
		return trace.reject("capture_changed");
	output = data->selected_capture;
	output.orientation = orientation_override == depth_orientation::normal || orientation_override == depth_orientation::reversed ?
		orientation_override : output.detected_orientation;
	const depth_calibration_key key { output.source_id, output.layout_epoch, output.width, output.height,
		output.x, output.y, output.active_width, output.active_height };
	if (const auto calibration = data->native_calibrations.find(output.source_id);
		calibration != data->native_calibrations.end() && calibration->second.matches(key))
	{
		const auto result = calibration->second.result(output.orientation);
		output.calibrated = result.calibrated;
		output.calibration_samples = result.samples;
		output.calibration_frame = result.sampled_frame;
		output.raw_anchor = result.raw_anchor;
		output.raw_gain = result.raw_gain;
	}
	// The runtime owns this view until finish_effects; the lookup cannot outlive
	// that callback. Do not hold the resource map lock while calling the device.
	output.shader_resource = data->selected_shader_resource;
	uint32_t color_width = 0, color_height = 0;
	runtime->get_screenshot_width_and_height(&color_width, &color_height);
	output.aligned_viewport_assumed = raw_alignment_assumed({ output.x, output.y, output.active_width, output.active_height, output.layout_epoch },
		output.width, output.height, color_width, color_height);
	output.resource = device->get_resource_from_view(output.shader_resource);
	if (output.resource == 0)
	{
		output = {};
		return trace.reject("missing_view_resource");
	}
	output.ready = true;
	if (device->get_api() == device_api::d3d12)
		output.command_queue = runtime->get_command_queue()->get_native();
	if (request_calibration)
	{
		data->native_ui_samples = output.calibration_samples;
		data->native_ui_state = output.calibrated ? native_depth_ui_state::ready :
			(output.calibration_samples >= 3 && output.orientation == depth_orientation::automatic ?
				native_depth_ui_state::waiting_orientation : native_depth_ui_state::calibrating);
	}
	return true;
}

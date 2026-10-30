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

#include "addon_lifetime.h"
#include "addon_config.h"
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
#include <cmath> // std::abs
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
#include "streamline_native_observer.h"
#include "native_resource_identity.h"
#include "game3d_slow_step.h"
#include "game3d_frame_record.h"
#include "depth_addon_test_access.h"
#include "capture_inputs.h"
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
#include "game3d_test_depth_events.h"
namespace depth_events = sunshine_game3d::test::depth_events;
#endif

using namespace reshade::api;

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

// Generic's candidate selection, defined in generic_selection.inl (included at
// the end of this file): the capture-region and shape rules the capture and
// raw-sample code use, and begin_depth_frame's candidate selection.
template <typename DepthInfo>
SUNSHINE_DEPTH_ADDON_INTERNAL capture_region effective_capture_region(const DepthInfo &info);
static bool raw_alignment_assumed(const capture_region &region, uint32_t source_width, uint32_t source_height,
	uint32_t color_width, uint32_t color_height);
SUNSHINE_DEPTH_ADDON_INTERNAL void observe_preserved_layout(depth_stencil_resource &info, const depth_stencil_frame_stats &stats);
template <typename DepthInfo>
SUNSHINE_DEPTH_ADDON_INTERNAL bool candidate_matches_shape(const DepthInfo &info, uint32_t width, uint32_t height);
static void select_depth_candidate(effect_runtime *runtime, device *const device,
	generic_depth_device_data *const device_data, generic_depth_data &data, const bool automatic,
	const std::optional<sunshine_depth::sample_result> &completed_sample,
	const sunshine_depth::depth_quality &completed_quality, const bool completed_other_raw_sample,
	resource &selected_depth_stencil, const depth_selection_resource *&selected_depth_stencil_info,
	uint32_t &frame_width, uint32_t &frame_height, uint64_t &native_present_index,
	depth_selection_snapshot &current_depth_stencil_resources,
	bool &any_recent_activity, bool &any_filtered, bool &any_warming);

// Generic depth's capture routing, defined in generic_capture.inl (included
// here): the FrameActivityTrace types, member views, capture records, the hold
// gates and hold, Present coverage, the end-of-frame copies and
// select_captured_depth.
#include "generic_capture.inl"

bool sunshine_depth::copy_selected_depth(effect_runtime *runtime, command_list *commands,
	const sunshine_streamline::depth_capture::packet &selection, resource destination, frame_depth &captured,
	sunshine_streamline::depth_capture::consumer_diagnostic *diagnostic)
{
	captured = {};
	constexpr uint32_t sampled_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
	return runtime && commands && destination.handle && sunshine_streamline::depth_capture::copy_current(
		commands->get_native(), selection, destination.handle, sampled_state, diagnostic,
		commands == runtime->get_command_queue()->get_immediate_command_list());
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

// Generic's challenger probe, defined in generic_probe.inl (included at the
// end of this file): releasing the challenger, preparing and sampling the
// probe target, and completing a raw sample.
static void release_challenger(effect_runtime *runtime, generic_depth_data &data, bool destroy_runtime);
static void prepare_probe_target(effect_runtime *runtime, command_list *cmd_list, generic_depth_data &data,
	generic_depth_device_data &device_data);
static void complete_raw_sample(effect_runtime *runtime, generic_depth_data &data,
	generic_depth_device_data &device_data, const sunshine_depth::sample_result &completed);
static void sample_probe_target(effect_runtime *runtime, command_list *cmd_list, generic_depth_data &data);

static void set_generic_capture_enabled(effect_runtime *runtime, generic_depth_data &data, bool enabled)
{
	if (data.generic_capture_enabled == enabled)
		return;
	auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	if (device_data == nullptr)
		return;
	data.generic_capture_enabled = enabled;
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
		data.held = false;
		data.shown = {};
		data.captures = {};
		data.selected_record = {};
		data.selected_capture = {};
		// Keep bounded selected/member allocations and CPU preference history for
		// fallback. Their GPU copies are gated below; no wait/reset or unsafe
		// immediate view/resource destruction is required for API handover.
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

// Generic depth's status reporting, defined in depth_panel.inl (included at
// the end of this file): the depth-format names, begin_depth_frame's status
// line and the overlay's settings panel.
static const char *const format_to_string(format value);
static void log_depth_status(generic_depth_data &data);
static void draw_settings_overlay(effect_runtime *runtime);

static void update_effect_runtime(effect_runtime *runtime)
{
	auto &data = *runtime->get_private_data<generic_depth_data>();

	// A rotating source binds only its stable copy (nothing while it has none).
	const resource_view view = data.rotating ? data.stable.view : data.selected_shader_resource;
	if (!data.effects_bound || view != data.effects_view)
	{
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
		depth_events::binding(runtime, depth_events::site_id::generic_bind, view != 0);
#endif
		const sunshine_game3d::slow_step step("generic depth binding");
		runtime->update_texture_bindings("DEPTH", view, view);
		data.effects_view = view;
		data.effects_bound = true;
	}

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
	// The value the call below writes, read the same way.
	depth_events::readiness(runtime, depth_events::site_id::generic_ready,
		view != 0 && (data.capture_ready || data.held) && (!data.rotating || data.stable_current));
#endif
	sunshine_streamline::provider::set_depth_ready(runtime, view != 0 && (data.capture_ready || data.held) && (!data.rotating || data.stable_current));
}

// The stable copy stays bound until this Present's binding update replaces it.
static void retire_stable_binding(generic_depth_data &data, const generic_depth_device_data &device_data)
{
	data.stable.destroy_after_frame = device_data.frame_index + generic_depth_device_data::backup_retirement_frames;
	data.retired_stable.push_back(data.stable);
	data.stable = {};
	data.stable_capture = {};
	data.stable_current = false;
}

// Copies the selected member's current backup into the stable binding (see
// generic_depth_data::rotating). source is in the shader-resource state.
static bool copy_stable_binding(effect_runtime *runtime, command_list *cmd_list, generic_depth_data &data,
	const generic_depth_device_data &device_data, resource source)
{
	device *const device = runtime->get_device();
	if (device->get_api() != device_api::d3d11 && device->get_api() != device_api::d3d12)
		return false;
	const resource_desc desc = device->get_resource_desc(source);
	auto &stable = data.stable;
	if (stable.texture != 0 && (stable.desc.texture.width != desc.texture.width ||
		stable.desc.texture.height != desc.texture.height || stable.desc.texture.format != desc.texture.format))
		retire_stable_binding(data, device_data);
	if (stable.texture == 0)
	{
		resource_desc stable_desc(desc.texture.width, desc.texture.height, 1, 1, desc.texture.format, 1,
			memory_heap::default_, resource_usage::shader_resource | resource_usage::copy_dest);
		if (!device->create_resource(stable_desc, nullptr, resource_usage::shader_resource, &stable.texture))
		{
			stable = {};
			return false;
		}
		const resource_view_desc view_desc(format_to_default_typed(desc.texture.format));
		if (!device->create_resource_view(stable.texture, resource_usage::shader_resource, view_desc, &stable.view))
		{
			device->destroy_resource(stable.texture); // Never recorded or bound.
			stable = {};
			return false;
		}
		device->set_resource_name(stable.texture, "Sunshine depth effects binding");
		stable.desc = desc;
	}
	cmd_list->barrier(source, resource_usage::shader_resource, resource_usage::copy_source);
	cmd_list->barrier(stable.texture, resource_usage::shader_resource, resource_usage::copy_dest);
	cmd_list->copy_resource(source, stable.texture);
	cmd_list->barrier(source, resource_usage::copy_source, resource_usage::shader_resource);
	cmd_list->barrier(stable.texture, resource_usage::copy_dest, resource_usage::shader_resource);
	return true;
}

static void destroy_stable_bindings(effect_runtime *runtime, generic_depth_data &data, uint64_t frame_index, bool all)
{
	device *const device = runtime->get_device();
	const auto destroy = [device](const generic_depth_data::stable_binding &entry) {
		if (entry.view != 0) device->destroy_resource_view(entry.view);
		if (entry.texture != 0) device->destroy_resource(entry.texture);
	};
	auto keep = data.retired_stable.begin();
	for (auto &entry : data.retired_stable)
	{
		if (all || frame_index >= entry.destroy_after_frame) { destroy(entry); continue; }
		*keep++ = entry;
	}
	data.retired_stable.erase(keep, data.retired_stable.end());
	if (all)
	{
		destroy(data.stable);
		data.stable = {};
	}
}

static void on_reload_effect_runtime(effect_runtime *runtime)
{
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
	// Runs before the exporter's handler of this event (registered earlier).
	const depth_events::phase_scope phase(depth_events::handler::reload);
#endif
	if (auto *data = runtime->get_private_data<generic_depth_data>(); data && data->native_driver)
	{
		// FX handles are recreated, but the native capture/scale owner is not.
		// Do not throw away a live capture epoch or retained real FG depth.
		sunshine_streamline::provider::reload_effect_bindings(runtime);
		data->effects_bound = false; // New effect descriptor tables.
		data->effects_read_depth_dirty = true;
		if (!sunshine_streamline::provider::selected(runtime)) update_effect_runtime(runtime);
		return;
	}
	sunshine_streamline::provider::reload(runtime);
	if (auto *data = runtime->get_private_data<generic_depth_data>())
	{
		data->effects_bound = false; // New effect descriptor tables.
		data->effects_read_depth_dirty = true;
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

// Fires before the state changes (user, preset or another add-on); begin_effects
// recounts. An enabled technique counts at once, before it first renders.
static bool on_set_technique_state(effect_runtime *runtime, effect_technique, bool enabled)
{
	if (auto *data = runtime->get_private_data<generic_depth_data>())
	{
		data->effects_read_depth = data->effects_read_depth || enabled;
		data->effects_read_depth_dirty = true;
	}
	return false;
}

// ReShade fires begin_effects only while its effects are loaded, so the
// technique list is complete here (it is empty while loading).
static void refresh_effects_read_depth(effect_runtime *runtime, generic_depth_data &data)
{
	if (!data.effects_read_depth_dirty) return;
	data.effects_read_depth_dirty = false;
	bool enabled = false;
	runtime->enumerate_techniques(nullptr, [&enabled](effect_runtime *owner, effect_technique technique) {
		enabled = enabled || owner->get_technique_state(technique);
	});
	data.effects_read_depth = enabled;
}

static void on_init_device(device *device)
{
	device->create_private_data<generic_depth_device_data>();
	if (device->get_api() == device_api::d3d12)
	{
		auto *data = device->get_private_data<generic_depth_device_data>();
		data->native_identity = sunshine_native_identity::device_cookie(reinterpret_cast<ID3D12Device *>(device->get_native()), true);
		if (!device->create_query_heap(query_type::timestamp, 1, &data->deferred_copy_marker))
			data->deferred_copy_marker = {};
	}
	// Before the first DLSS feature is created, outside DLL loader lock. This
	// catches the game's linked NGX exports even if the runtime DLL loads later.
	if (device->get_api() == device_api::d3d12)
		sunshine_upscaler_trace::poll();
	// The content sampler's shader compiles off the first sample's Present.
	sunshine_depth::warm();

	reshade::get_config_value(nullptr, "DEPTH", "DisableINTZ", s_disable_intz);
}
static void on_destroy_device(device *device)
{
	const generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();

	// Destroy any remaining resources
	for (const depth_stencil_backup &backup : device_data->depth_stencil_backups)
	{
		device->destroy_resource(backup.backup_texture);
	}
	if (device_data->deferred_copy_marker != 0)
		device->destroy_query_heap(device_data->deferred_copy_marker);

	device->destroy_private_data<generic_depth_device_data>();

	// The capture owner's sources and snapshots of this device go while ReShade
	// still has it (each of its resources calls into it as it is destroyed).
	// Their tracked depth-stencil entries were dropped with the device data.
	if (device->get_api() == device_api::d3d12)
	{
		char line[320]{};
		const bool released = capture_inputs::release_device(device, line, sizeof(line));
		if (line[0])
			sunshine_log::message(released ? reshade::log::level::info : reshade::log::level::warning, line);
	}
}

static void on_init_command_list(command_list *cmd_list)
{
	cmd_list->create_private_data<state_tracking>(false);
	capture_inputs::list_event(cmd_list, sunshine_streamline::depth_capture::list_event::created);
	capture_inputs::observe_command(cmd_list);
}
static void on_destroy_command_list(command_list *cmd_list)
{
	capture_inputs::list_event(cmd_list, sunshine_streamline::depth_capture::list_event::destroyed); // Retires its recording.
	cmd_list->destroy_private_data<state_tracking>();
}

static void on_init_command_queue(command_queue *cmd_queue)
{
	cmd_queue->create_private_data<state_tracking>(true);
	capture_inputs::observe_queue(cmd_queue);

	if ((cmd_queue->get_type() & command_queue_type::graphics) == 0)
		return;

	generic_depth_device_data *const device_data = cmd_queue->get_device()->get_private_data<generic_depth_device_data>();
	device_data->queues.push_back(cmd_queue);
}
static void on_destroy_command_queue(command_queue *cmd_queue)
{
	capture_inputs::retire_queue(cmd_queue);
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
	sunshine_game3d::frame_record_store::initialize(runtime);
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
		retire_member_views(runtime, data, *device_data, true);
	}
	if (data.stable.texture != 0 || !data.retired_stable.empty())
	{
		// Effects of earlier Presents may still read these copies.
		runtime->get_command_queue()->wait_idle();
		destroy_stable_bindings(runtime, data, 0, true);
	}

	runtime->destroy_private_data<generic_depth_data>();
	sunshine_streamline::provider::destroy(runtime);
	sunshine_game3d::frame_record_store::destroy(runtime);
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
			sunshine_log::message(reshade::log::level::warning, "Replacing high bit depth depth-stencil format with a lower bit depth format");
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
static void on_init_resource(device *device, const resource_desc &desc, const subresource_data *, resource_usage initial_state, resource resource)
{
	if (desc.type != resource_type::surface && desc.type != resource_type::texture_2d && (desc.type != resource_type::texture_3d || desc.texture.depth_or_layers > 1))
		return;

	if ((desc.usage & resource_usage::depth_stencil) == 0)
		return;

	// The end-of-frame copy needs its state proven (record_preserved); the
	// capture owner tracks it from this creation state.
	if (device->get_api() == device_api::d3d12)
		capture_inputs::resource_created(resource, initial_state);

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
		// The capture owner's proven-state table forgets it (resource_created
		// tracked it under this native identity).
		if (it->second.api_identity)
			capture_inputs::resource_destroyed(it->second.identity);
		device_data->depth_stencil_resources.erase(it);

		// A backup resource is always created in D3D12 and Vulkan, so to find out if an effect runtime references this depth-stencil resource, can simply check if a backup resource was created for it
		if (device_data->find_depth_stencil_backup(resource) != nullptr)
		{
			lock.unlock();

			sunshine_log::message(reshade::log::level::warning, "A depth-stencil resource was destroyed while still in use.");

			// This is bad ... the resource may still be in use by an effect on the GPU and destroying it would crash it
			// Try to mitigate that somehow by delaying this thread a little to hopefully give the GPU enough time to catch up before the resource memory is deallocated
			Sleep(500);
		}
	}
}

// Generic's command recording, defined in generic_recording.inl (included at
// the end of this file): the recording events register_depth_events
// registers, and the transfer lineage on_present resolves.
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_draw(command_list *cmd_list, uint32_t vertices, uint32_t instances, uint32_t, uint32_t);
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_draw_indexed(command_list *cmd_list, uint32_t indices, uint32_t instances, uint32_t, int32_t, uint32_t);
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_draw_indirect(command_list *cmd_list, indirect_command type, resource, uint64_t, uint32_t draw_count, uint32_t);
static void on_bind_viewport(command_list *cmd_list, uint32_t first, uint32_t count, const viewport *viewport);
static void on_bind_depth_stencil(command_list *cmd_list, uint32_t, const resource_view *, resource_view depth_stencil_view);
static bool on_clear_depth_stencil(command_list *cmd_list, resource_view dsv, const float *depth, const uint8_t *, uint32_t rect_count, const rect *);
static bool on_begin_render_pass_with_depth_stencil(command_list *cmd_list, uint32_t, const render_pass_render_target_desc *, const render_pass_depth_stencil_desc *depth_stencil_desc, render_pass_flags);
static bool on_end_render_pass(command_list *cmd_list);
SUNSHINE_DEPTH_ADDON_INTERNAL void resolve_transfer_lineage(state_tracking &state, const generic_depth_device_data &device_data);
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_dispatch_mesh(command_list *cmd_list, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_copy_resource(command_list *cmd_list, resource source, resource dest);
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_copy_texture_region(command_list *cmd_list, resource source, uint32_t source_subresource,
	const subresource_box *source_box, resource dest, uint32_t subresource, const subresource_box *dest_box, filter_mode);
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_copy_buffer_to_texture(command_list *cmd_list, resource, uint64_t, uint32_t, uint32_t,
	resource dest, uint32_t subresource, const subresource_box *dest_box);
SUNSHINE_DEPTH_ADDON_INTERNAL bool on_resolve_texture_region(command_list *cmd_list, resource source, uint32_t source_subresource,
	const subresource_box *source_box, resource dest, uint32_t subresource, uint32_t dest_x, uint32_t dest_y, uint32_t dest_z, format);
static void on_content_barrier(command_list *, uint32_t, const resource *, const resource_usage *, const resource_usage *);
static void on_reset(command_list *cmd_list);
static void on_close(command_list *cmd_list);
static void on_execute_primary(command_queue *queue, command_list *cmd_list);
static void on_execute_secondary(command_list *cmd_list, command_list *secondary_cmd_list);

// Submits the copies present_deferred recorded natively on the queue's
// immediate list before this Present's effects read them. ReShade submits its
// immediate list only after a command recorded through its own API
// (d3d12::command_list_immediate_impl::flush), so one timestamp query, written
// and never read, marks it first: without it the list would be submitted
// with the effects, after this Present's depth was chosen, so no copy is
// made (on_present) and the deferred tags lapse.
static void submit_deferred_copies(command_queue *queue, command_list *immediate, query_heap marker)
{
	immediate->end_query(marker, query_type::timestamp, 0);
	// ReShade's Signals of its own fences in this flush never wait on the
	// queue-order lock (own_flush_scope).
	const sunshine_streamline::native_observer::own_flush_scope own;
	queue->flush_immediate_command_list();
}

static void on_present(command_queue *presenting_queue, swapchain *swapchain, const rect *, const rect *, uint32_t, const rect *)
{
	// The native tag's UntilPresent lifetime ends here. Already copied packets
	// remain owned and usable by effects; tags supplied after this boundary keep
	// their new authority, including concurrent next-frame tagging. First, the
	// copies a tag call deferred to its frame's Present (owner ruling R2) are
	// recorded on this queue's immediate list and submitted ahead of it.
	if (presenting_queue->get_device()->get_api() == device_api::d3d12)
	{
		const query_heap marker = presenting_queue->get_device()->get_private_data<generic_depth_device_data>()->deferred_copy_marker;
		if (command_list *const immediate = presenting_queue->get_immediate_command_list();
			immediate != nullptr && marker != 0 && sunshine_streamline::present_deferred(presenting_queue->get_native(), immediate->get_native()))
			submit_deferred_copies(presenting_queue, immediate, marker);
		capture_inputs::present(presenting_queue);
	}
	sunshine_depth::begin_present(swapchain);
	device *const device = swapchain->get_device();
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();

	// Declared before the lock, so its destructor emits only after unlocking.
	std::optional<activity_present_trace> trace;
	if (s_frame_activity_trace) trace.emplace(swapchain, presenting_queue);
	std::unique_lock<std::shared_mutex> lock(s_mutex);
	++device_data->native_present_index;
	const auto demand = tracking_demand(*device_data);
	if (trace) trace->begin(*device_data);

	state_tracking queue_state(true);
	// Copies are judged against their sources in each queue's execution order.
	const bool content_selection = device->get_api() == device_api::d3d11 || device->get_api() == device_api::d3d12;
	// Merge state from all graphics queues
	for (command_queue *const queue : device_data->queues)
	{
		auto &state = *queue->get_private_data<state_tracking>();
		if (content_selection) resolve_transfer_lineage(state, *device_data);
		if (trace) trace->add(queue, state, *device_data);
		queue_state.merge(state, demand);

		state.reset_on_present();
	}

	// Only update device list if there are any depth-stencils, otherwise this may be a second present call (at which point 'reset_on_present' already cleared out the queue list in the first present call)
	if (queue_state.stats_per_used_depth_stencil.empty())
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
			info.orientation_evidence.observe(device_data->frame_index, frame_stats.clear_zero, frame_stats.clear_one, frame_stats.clear_other);

			// Save to current list of depth-stencils on the device
			info.observe_frame(frame_stats, device_data->frame_index);
		}
	}

	// Destroy resources that were enqueued for delayed destruction and have reached the targeted number of passed frames.
	// Erase them under the lock and destroy them after it: game-thread depth clears wait for this exclusive section.
	std::vector<resource> expired;
	for (auto it = device_data->depth_stencil_backups.begin(); it != device_data->depth_stencil_backups.end();)
	{
		if (device_data->frame_index >= it->destroy_after_frame)
		{
			assert(it->references == 0);

			expired.push_back(it->backup_texture);
			it = device_data->depth_stencil_backups.erase(it);
			continue;
		}

		++it;
	}
	lock.unlock();
	for (const resource texture : expired)
		device->destroy_resource(texture);
}

// The frame record's Generic row (game3d_frame_record.h), as this Present's
// selection left it; nothing reads it.
static void record_generic_row(effect_runtime *runtime, const generic_depth_data &data, uint64_t present, bool api_owner)
{
	auto *const record = sunshine_game3d::frame_record_store::writable(runtime);
	if (record == nullptr)
		return;
	using sunshine_game3d::source_state;
	sunshine_game3d::generic_depth_decision decision;
	decision.api_owner = api_owner;
	if (!api_owner)
	{
		switch (data.binding_state)
		{
		case depth_binding_state::ready: decision.state = source_state::ready; break;
		case depth_binding_state::waiting_capture: decision.state = source_state::pending; break;
		case depth_binding_state::binding_failed:
		case depth_binding_state::filtered: decision.state = source_state::refused; break;
		default: decision.state = source_state::absent; break;
		}
		decision.reason = binding_state_name(data.binding_state);
		decision.identity = data.selected_identity;
	}
	sunshine_game3d::record_generic_depth(*record, decision, present, GetTickCount64());
}

static void begin_depth_frame(effect_runtime *runtime, command_list *cmd_list)
{
	device *const device = runtime->get_device();
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();

	if (device_data == nullptr)
		return;

	auto &data = *runtime->get_private_data<generic_depth_data>();
	// D3D11/12 select by content; other APIs keep ReShade's draw-statistics picker.
	const bool automatic = device->get_api() == device_api::d3d11 || device->get_api() == device_api::d3d12;
	// Native mode runs this every Present, also while ReShade skips its effects
	// (globally disabled, none loaded) and so never recounts the technique flag.
	const bool effects_may_read_depth = data.effects_read_depth && runtime->get_effects_state();
	data.native_access_open = false;
	using sunshine_game3d::slow_step;
	auto completed_sample = [&] { const slow_step step("depth sample poll"); return sunshine_depth::poll(runtime); }();
	if (completed_sample && [&] { const slow_step step("Streamline sample completion");
			return sunshine_streamline::provider::complete(runtime, *completed_sample); }())
		completed_sample.reset();
	// Point classification serves candidate preference only. Established-source
	// scene controls independently consume authenticated full-image statistics.
	const auto completed_quality = [&] {
		const slow_step step("depth sample analysis");
		return completed_sample && completed_sample->valid ?
			sunshine_depth::analyze_depth(completed_sample->values.data(), completed_sample->values.size(), completed_sample->width, completed_sample->height) :
			sunshine_depth::depth_quality {};
	}();
	const bool completed_other_raw_sample = completed_sample && completed_sample->token != data.challenger_id &&
		completed_sample->capture_id != 0 && completed_sample->capture_id == data.raw_pending_capture_id && data.raw_pending_metadata.depth_ready;
	if (completed_sample)
	{
		const slow_step step("raw depth sample completion");
		complete_raw_sample(runtime, data, *device_data, *completed_sample);
		release_sample_reference(runtime, data);
	}
	const bool was_streamline = sunshine_streamline::provider::selected(runtime);
	const bool api_selected = [&] {
		const slow_step step("Streamline depth acquisition");
		return sunshine_streamline::provider::begin(runtime, cmd_list, device_data->native_present_index,
			automatic && s_streamline_source_events);
	}();
	// A native API snapshot needs no Generic work. Fallback resumes capture
	// before entering the selector.
	{
		const slow_step step("generic capture switch");
		set_generic_capture_enabled(runtime, data, !api_selected);
	}
	if (api_selected)
	{
		if (!was_streamline)
		{
			const slow_step step("generic challenger release");
			release_challenger(runtime, data, true);
		}
		{
			// Rotation members are bound only while selected. While API depth owns
			// the pass they retire on the usual delay, releasing their game depth
			// references and backups; the selected view stays for fallback.
			const slow_step step("generic member retirement");
			for (auto &member : data.member_views)
				if (member.view != 0 && !member.retire_after && member.view != data.selected_shader_resource)
					member.retire_after = device_data->frame_index + generic_depth_device_data::backup_retirement_frames;
			retire_member_views(runtime, data, *device_data);
		}
		data.effects_bound = false; // The API provider owns the DEPTH binding.
		data.held = false;
		data.shown = {};
		data.native_access_present = device_data->native_present_index;
		data.native_access_open = true;
		record_generic_row(runtime, data, device_data->native_present_index, true);
		return;
	}
	if (was_streamline)
	{
		const slow_step step("generic depth rebinding");
		update_effect_runtime(runtime);
	}
	// No enabled technique can read the stable copy; this Present unbinds it.
	if (!effects_may_read_depth && data.stable.texture != 0)
		retire_stable_binding(data, *device_data);
	if (!data.retired_stable.empty())
		destroy_stable_bindings(runtime, data, device_data->frame_index, false);

	// Filled by select_depth_candidate (generic_selection.inl).
	resource selected_depth_stencil;
	const depth_selection_resource *selected_depth_stencil_info;
	uint32_t frame_width, frame_height;
	uint64_t native_present_index;
	depth_selection_snapshot current_depth_stencil_resources;
	bool any_recent_activity, any_filtered, any_warming;
	select_depth_candidate(runtime, device, device_data, data, automatic, completed_sample, completed_quality,
		completed_other_raw_sample, selected_depth_stencil, selected_depth_stencil_info, frame_width, frame_height,
		native_present_index, current_depth_stencil_resources, any_recent_activity, any_filtered, any_warming);

	const resource_view prev_shader_resource = data.selected_shader_resource;
	data.stable_current = false; // Set again below by this Present's stable copy.
	auto *member_binding = select_captured_depth(runtime, cmd_list, data, *device_data, current_depth_stencil_resources,
		automatic, selected_depth_stencil, selected_depth_stencil_info);
	// Judged on the Present that advanced the depth frame, from its own work. A
	// later Present without depth work can hold only what that one showed.
	data.hold_verdict = device_data->native_depth_present_index == native_present_index ?
		judge_hold_gate(current_depth_stencil_resources, data.captures, data.shown.record.identity, device_data->frame_index,
			frame_width, frame_height, device->check_capability(device_caps::resolve_depth_stencil),
			data.selection.active_grace_frames()) : hold_gate::open;
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
	data.held = false;
	data.selected_record = {};
	data.selected_capture = {};
	data.binding_state = current_depth_stencil_resources.empty() ? depth_binding_state::waiting_game :
		(any_filtered && any_recent_activity ? depth_binding_state::filtered : depth_binding_state::waiting_activity);
	data.binding_detail = current_depth_stencil_resources.empty() ? "Waiting for the game to create a depth buffer." :
		(any_filtered && any_recent_activity ? "Depth buffers exist, but none meets the aspect or sample filters." :
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
			(data.raw_scene_requested && depth_stencil_backup == nullptr))
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
			if (automatic || data.raw_scene_requested || (selected_depth_stencil_info->desc.usage & resource_usage::shader_resource) == 0 || selected_depth_stencil_info->desc.texture.samples > 1 || (api == device_api::d3d12 || api == device_api::vulkan))
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
		// Clears alone cannot make a source ready; raster, mesh and copied depth can.
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
		// A source selected only to be held (routing chose none: no qualified
		// current copy) never shows its own copy; a later copy into its backup
		// also ends the hold.
		data.capture_ready = !data.hold_only &&
			captured.record.renderable(device_data->native_present_index, device_data->frame_index, data.runtime_epoch);
		// After all of this Present's copies: a later copy into the held backup ends the hold.
		data.held = !data.capture_ready && hold_shown_capture(runtime, data, *device_data, frame_width, frame_height);
		data.selected_record = data.held ? data.shown.record : captured.record;
		data.selected_capture = data.held ? data.shown.metadata : captured.metadata;
		if (data.held) data.selected_capture.reused_depth = true;
		if (data.capture_ready)
			++(captured.end_of_frame ? data.end_of_frame_copy_frames : data.boundary_copy_frames);
		if (data.held) ++data.held_frames;
		if (data.rotating && effects_may_read_depth && data.using_backup_texture && (data.capture_ready || data.held))
		{
			// Only a new ready capture is copied (a held one is the copy already
			// there). Its sampled texture is the selected view's backup, in the
			// shader-resource state.
			const auto &shown_record = data.selected_record;
			const std::array<uint64_t, 4> key = { shown_record.identity, shown_record.sampled,
				shown_record.frame, shown_record.present };
			data.stable_current = data.stable.view != 0 && key == data.stable_capture;
			if (!data.stable_current)
			{
				const slow_step step("generic stable depth copy");
				data.stable_current = copy_stable_binding(runtime, cmd_list, data, *device_data, resource { shown_record.sampled });
				data.stable_capture = data.stable_current ? key : std::array<uint64_t, 4> {};
			}
		}
		for (unsigned i = 0; i != data.captures.count; ++i)
			if (data.captures.records[i].identity == data.selected_identity)
			{
				data.captures.records[i] = captured.record;
			}
		data.binding_state = data.capture_ready || data.held ? depth_binding_state::ready : depth_binding_state::waiting_capture;
		data.binding_detail = data.capture_ready ? "Depth is bound and its current frame is available." :
			data.held ? "Depth is bound; this Present shows the last current frame again, within the source age." :
			data.hold_verdict == hold_gate::cleared ? "The game cleared its depth without drawing (a screen without its scene); nothing is held." :
			data.hold_verdict == hold_gate::cut ? "Another depth buffer started drawing a scene; nothing is held until a current copy qualifies." :
			(source_active ? "Current buffer is selected; waiting for a valid end-of-frame copy." : "Current buffer is selected but inactive; waiting for it to render again.");
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
	// What this Present showed, for the next Present's hold. A hold keeps the
	// original tick and serial; a Present without Generic depth clears it.
	if (data.capture_ready)
	{
		data.shown = {};
		data.shown.record = data.selected_record;
		data.shown.metadata = data.selected_capture;
		data.shown.tick = GetTickCount64();
		data.shown.anchor = data.anchor_identity;
		data.shown.color_width = frame_width;
		data.shown.color_height = frame_height;
		const std::shared_lock<std::shared_mutex> serial_lock(s_mutex);
		for (const auto &backup : device_data->depth_stencil_backups)
			if (backup.depth_stencil == data.selected_depth_stencil && backup.backup_texture.handle == data.selected_record.sampled)
				data.shown.content_serial = backup.content_serial;
	}
	else if (!data.held)
		data.shown = {};
	// Evidence for later choices, once per Present that advanced the depth frame.
	if (automatic && device_data->native_depth_present_index == native_present_index)
		observe_present_coverage(data, device_data->frame_index, current_depth_stencil_resources);

	// Rebinds only when the effects view changed; a readiness flip only
	// publishes readiness.
	update_effect_runtime(runtime);
	if (prev_shader_resource != data.selected_shader_resource && !owns_member_view(data, prev_shader_resource))
		device->destroy_resource_view(prev_shader_resource);
	log_depth_status(data); // The status line (depth_panel.inl).
	prepare_probe_target(runtime, cmd_list, data, *device_data);
	data.native_access_present = native_present_index;
	data.native_access_open = true;
	record_generic_row(runtime, data, native_present_index, false);
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
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
	// Runs before the exporter's handler of this event (registered earlier).
	const depth_events::phase_scope phase(depth_events::handler::begin_effects);
#endif
	auto *data = runtime->get_private_data<generic_depth_data>();
	if (!data) return;
	refresh_effects_read_depth(runtime, *data);
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

// The content sampler's private list holds only the add-on's own copies, so
// its submission is not the game's work: the submission hook does not report
// it (it would leave every proven state unproven, proven_state.h).
static void finish_sampler_present(command_queue *queue, swapchain *swapchain)
{
	const sunshine_streamline::native_observer::suppression_scope add_on_work;
	sunshine_depth::finish_present(queue, swapchain);
}

static void register_depth_events()
{
	s_streamline_source_events = sunshine_streamline::source_enabled() || sunshine_upscaler_trace::capture_enabled();
	sunshine_streamline::depth_capture::initialize(true); // Shared D3D12 capture also serves Generic fallback.
	s_frame_activity_trace = false;
	s_frame_activity_trace = sunshine_game3d::addon_config::frame_activity_trace();
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
	reshade::register_event<reshade::addon_event::close_command_list>(sunshine_addon_lifetime::guarded<on_close>);
	reshade::register_event<reshade::addon_event::bind_viewports>(sunshine_addon_lifetime::guarded<on_bind_viewport>);
	reshade::register_event<reshade::addon_event::begin_render_pass>(sunshine_addon_lifetime::guarded<on_begin_render_pass_with_depth_stencil>);
	reshade::register_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(sunshine_addon_lifetime::guarded<on_bind_depth_stencil>);
	reshade::register_event<reshade::addon_event::clear_depth_stencil_view>(sunshine_addon_lifetime::guarded<on_clear_depth_stencil>);

	reshade::register_event<reshade::addon_event::reset_command_list>(sunshine_addon_lifetime::guarded<on_reset>);
	reshade::register_event<reshade::addon_event::execute_command_list>(sunshine_addon_lifetime::guarded<on_execute_primary>);
	reshade::register_event<reshade::addon_event::execute_secondary_command_list>(sunshine_addon_lifetime::guarded<on_execute_secondary>);

	reshade::register_event<reshade::addon_event::present>(sunshine_addon_lifetime::guarded<on_present>);
	reshade::register_event<reshade::addon_event::finish_present>(sunshine_addon_lifetime::guarded<finish_sampler_present>);

	reshade::register_event<reshade::addon_event::reshade_begin_effects>(sunshine_addon_lifetime::guarded<on_begin_render_effects>);
	reshade::register_event<reshade::addon_event::reshade_finish_effects>(sunshine_addon_lifetime::guarded<on_finish_render_effects>);
	// Need to set texture binding again after reloading
	reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(sunshine_addon_lifetime::guarded<on_reload_effect_runtime>);
	reshade::register_event<reshade::addon_event::reshade_set_technique_state>(sunshine_addon_lifetime::guarded<on_set_technique_state>);
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
	reshade::unregister_event<reshade::addon_event::close_command_list>(sunshine_addon_lifetime::guarded<on_close>);
	reshade::unregister_event<reshade::addon_event::bind_viewports>(sunshine_addon_lifetime::guarded<on_bind_viewport>);
	reshade::unregister_event<reshade::addon_event::begin_render_pass>(sunshine_addon_lifetime::guarded<on_begin_render_pass_with_depth_stencil>);
	reshade::unregister_event<reshade::addon_event::bind_render_targets_and_depth_stencil>(sunshine_addon_lifetime::guarded<on_bind_depth_stencil>);
	reshade::unregister_event<reshade::addon_event::clear_depth_stencil_view>(sunshine_addon_lifetime::guarded<on_clear_depth_stencil>);

	reshade::unregister_event<reshade::addon_event::reset_command_list>(sunshine_addon_lifetime::guarded<on_reset>);
	reshade::unregister_event<reshade::addon_event::execute_command_list>(sunshine_addon_lifetime::guarded<on_execute_primary>);
	reshade::unregister_event<reshade::addon_event::execute_secondary_command_list>(sunshine_addon_lifetime::guarded<on_execute_secondary>);

	reshade::unregister_event<reshade::addon_event::present>(sunshine_addon_lifetime::guarded<on_present>);
	reshade::unregister_event<reshade::addon_event::finish_present>(sunshine_addon_lifetime::guarded<finish_sampler_present>);

	reshade::unregister_event<reshade::addon_event::reshade_begin_effects>(sunshine_addon_lifetime::guarded<on_begin_render_effects>);
	reshade::unregister_event<reshade::addon_event::reshade_finish_effects>(sunshine_addon_lifetime::guarded<on_finish_render_effects>);
	reshade::unregister_event<reshade::addon_event::reshade_reloaded_effects>(sunshine_addon_lifetime::guarded<on_reload_effect_runtime>);
	reshade::unregister_event<reshade::addon_event::reshade_set_technique_state>(sunshine_addon_lifetime::guarded<on_set_technique_state>);
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
		sunshine_log::message(reshade::log::level::error, s_status_message);
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
			sunshine_log::message(reshade::log::level::error, s_status_message);
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
		sunshine_log::message(reshade::log::level::warning, s_status_message);
		return false;
	}
	if (legacy == legacy_depth_state::unknown)
	{
		s_status_message = "Depth selection could not verify the loaded add-ons. Restart the game after removing old Sunshine Depth add-ons; export remains available for this launch.";
		sunshine_log::message(reshade::log::level::warning, s_status_message);
		return false;
	}
	if (legacy == legacy_depth_state::present)
	{
		s_status_message = "The old Sunshine Depth add-on is already loaded, possibly under a renamed file. Close the game, remove that old add-on, and restart. Keep only the Sunshine 3D add-on; automatic selection will then use its integrated depth selector.";
		sunshine_log::message(reshade::log::level::warning, s_status_message);
		return false;
	}
	register_depth_events();
	s_registered = true;
	s_status_message = "Depth selection is ready.";
	sunshine_log::message(reshade::log::level::info, "Sunshine 3D depth selector initialized.");
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
// The test add-on's exports (test_depth_addon_exports.cpp) use these with
// both the recorded and the snapshot resource rows.
template capture_region effective_capture_region(const depth_stencil_resource &);
template capture_region effective_capture_region(const depth_selection_resource &);
template bool candidate_matches_shape(const depth_stencil_resource &, uint32_t, uint32_t);
template bool candidate_matches_shape(const depth_selection_resource &, uint32_t, uint32_t);
template void observe_candidate(generic_depth_data &, const depth_stencil_resource &, uint32_t, uint32_t);
template void observe_candidate(generic_depth_data &, const depth_selection_resource &, uint32_t, uint32_t);
#endif

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
	if (!data->capture_ready && !data->held) return trace.reject("capture_unavailable");
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
	uint32_t frame_width = 0, frame_height = 0;
	runtime->get_screenshot_width_and_height(&frame_width, &frame_height);
	if (data->native_access_present != device_data->native_present_index ||
		!(data->held ? hold_shown_capture(runtime, *data, *device_data, frame_width, frame_height, false) :
			validate_capture_record(runtime, *data, *device_data, data->selected_record)))
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

#include "generic_recording.inl"
#include "generic_probe.inl"
#include "generic_selection.inl"
#include "depth_panel.inl"
#include "generic_raw_evidence.inl"

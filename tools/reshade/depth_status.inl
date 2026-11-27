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

// Generic depth's status reporting, a part of depth_addon.cpp (included at
// its end, in the same translation unit): log_depth_status, begin_depth_frame's
// rate-limited "Sunshine 3D depth:" line, the depth-format names, and the
// read-only depth rows the overlay's settings panel (game3d_controls.cpp)
// draws, with the overlay's and the test add-on's entries to that panel.
// Declared in depth_addon.cpp.
#pragma once

static void log_depth_status(generic_depth_data &data)
{
	const bool selection_changed = data.last_logged_id != data.anchor_identity;
	const bool binding_changed = data.last_logged_state != data.binding_state;
	// The game may alternate its tagged allocation every present. Ordinary
	// source changes share the status rate limit; binding errors remain
	// immediate. Logging must never turn physical rotation into per-frame I/O.
	// The copy counts alone repeat the line at most every 10 s.
	const uint64_t copy_frames = data.boundary_copy_frames + data.end_of_frame_copy_frames + data.held_frames + data.end_of_frame_refused;
	if (data.status_log.due(sunshine_game3d::clock::game_now(), selection_changed || binding_changed,
		binding_changed && data.binding_state == depth_binding_state::binding_failed,
		copy_frames != data.logged_copy_frames))
	{
		char message[1024];
		const auto proven = sunshine_streamline::depth_capture::proven_counts();
		std::snprintf(message, std::size(message),
			"Sunshine 3D depth: %s; source=%llu %ux%u %s; %s; %s; ready_copies boundary=%llu end_of_frame=%llu held=%llu; "
			"end_of_frame_refused=%llu last=%s; proven tracked=%llu unobserved={unrecorded=%llu unreported=%llu invalidated=%llu "
			"missed_reset=%llu uncovered=%llu} changed_copies=%llu latched=%llu%s earlier_layout={copied=%llu refused=%llu} "
			"copy_out={made=%llu ring_busy=%llu failed=%llu predicted=%llu taken=%llu unstored=%llu released=%llu skipped=%llu "
			"lost=%llu reader_moves=%llu move_unreadable=%llu}",
			binding_state_name(data.binding_state), static_cast<unsigned long long>(data.selected_identity),
			data.selected_desc.texture.width, data.selected_desc.texture.height, format_to_string(data.selected_desc.texture.format),
			data.selection_reason, data.binding_detail, static_cast<unsigned long long>(data.boundary_copy_frames),
			static_cast<unsigned long long>(data.end_of_frame_copy_frames), static_cast<unsigned long long>(data.held_frames),
			static_cast<unsigned long long>(data.end_of_frame_refused),
			sunshine_streamline::depth_capture::name(data.end_of_frame_refusal), static_cast<unsigned long long>(proven.tracked),
			static_cast<unsigned long long>(proven.unrecorded), static_cast<unsigned long long>(proven.unreported),
			static_cast<unsigned long long>(proven.invalidated), static_cast<unsigned long long>(proven.missed_reset),
			static_cast<unsigned long long>(proven.uncovered), static_cast<unsigned long long>(proven.changed_copies),
			static_cast<unsigned long long>(proven.latched), proven.latch_lost ? " latch_lost" : "",
			static_cast<unsigned long long>(proven.earlier_copied), static_cast<unsigned long long>(proven.earlier_refused),
			static_cast<unsigned long long>(proven.copied_out), static_cast<unsigned long long>(proven.copy_out_busy),
			static_cast<unsigned long long>(proven.copy_out_failed), static_cast<unsigned long long>(proven.copy_out_predicted),
			static_cast<unsigned long long>(proven.copy_out_taken), static_cast<unsigned long long>(proven.copy_out_unstored),
			static_cast<unsigned long long>(proven.copy_out_released), static_cast<unsigned long long>(proven.copy_out_skipped),
			static_cast<unsigned long long>(proven.copy_out_lost),
			static_cast<unsigned long long>(proven.reader_moves), static_cast<unsigned long long>(proven.reader_move_unreadable));
		sunshine_log::message(data.binding_state == depth_binding_state::binding_failed ? reshade::log::level::error : reshade::log::level::info, message);
		data.last_logged_id = data.anchor_identity;
		data.last_logged_state = data.binding_state;
		data.logged_copy_frames = copy_frames;
	}
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

	default:
		return "     ";
	}
}

SUNSHINE_DEPTH_ADDON_INTERNAL sunshine_streamline::provider::source_status describe_streamline_source(effect_runtime *runtime)
{
	const auto *device_data = runtime->get_device()->get_private_data<generic_depth_device_data>();
	uint64_t present;
	{
		const std::shared_lock<std::shared_mutex> status_lock(s_mutex);
		present = device_data ? device_data->native_present_index : std::numeric_limits<uint64_t>::max();
	}
	return sunshine_streamline::provider::describe(runtime, present);
}

// The overlay's depth rows, read for game3d_controls.cpp's settings panel when
// it draws them: the selector's setup state, the API source's description and
// Generic's binding.
static sunshine_game3d::depth_panel_status read_depth_panel_status(effect_runtime *runtime)
{
	using panel_status = sunshine_game3d::depth_panel_status;
	panel_status out;
	if (!s_registered)
	{
		out.availability = panel_status::state::not_registered;
		out.message = s_status_message;
		return out;
	}
	if (runtime != nullptr && !supported_api(runtime->get_device()))
	{
		out.availability = panel_status::state::unsupported_api;
		return out;
	}
	if (runtime == nullptr || runtime->get_private_data<generic_depth_data>() == nullptr)
	{
		out.availability = panel_status::state::waiting_for_runtime;
		return out;
	}
	auto &data = *runtime->get_private_data<generic_depth_data>();
	device *const device = runtime->get_device();
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();
	const auto provided = describe_streamline_source(runtime);
	out.availability = panel_status::state::ready;
	out.game_provided = provided.selected;
	out.ngx = provided.provider == sunshine_scene_depth::provider_kind::ngx;
	out.provided_last_valid = provided.last_valid.resource != 0;
	out.provided_width = provided.last_valid.width;
	out.provided_height = provided.last_valid.height;
	out.presentations = provided.presentations;
	{
		const std::shared_lock<std::shared_mutex> status_lock(s_mutex);
		out.generic_current = !provided.selected && device_data && (data.capture_ready || data.held) && data.selected_shader_resource != 0 &&
			data.selected_depth_stencil != 0 && data.native_access_present == device_data->native_present_index;
	}
	out.generic_buffer = data.selected_shader_resource != 0 && data.selected_depth_stencil != 0;
	out.generic_width = data.selected_desc.texture.width;
	out.generic_height = data.selected_desc.texture.height;
	out.generic_format = format_to_string(data.selected_desc.texture.format);
	out.binding_ready = data.binding_state == depth_binding_state::ready;
	out.binding_detail = data.binding_detail;
	return out;
}

static void draw_settings_overlay(effect_runtime *runtime)
{
	sunshine_game3d::draw_settings_panel(runtime, read_depth_panel_status);
}

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
void sunshine_game3d_test_draw_production_panel(effect_runtime *runtime, unsigned mode)
{
	// Exercise the actual production tabs without changing capture or settings.
	sunshine_game3d::draw_settings_panel(runtime, read_depth_panel_status, static_cast<int>(mode) - 1);
}
#endif

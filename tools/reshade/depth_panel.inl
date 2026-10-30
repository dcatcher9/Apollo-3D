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
// overlay's settings panel with its Status, Troubleshooting and Calibration
// tabs (and the test add-on's entry to them). Declared in depth_addon.cpp.
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
	if (data.status_log.due(GetTickCount64(), selection_changed || binding_changed,
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
			"reader_moves=%llu move_unreadable=%llu}",
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
	case format::intz:
		return "INTZ ";
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

static void draw_depth_panel(effect_runtime *runtime, bool troubleshooting)
{
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
	auto &data = *runtime->get_private_data<generic_depth_data>();
	device *const device = runtime->get_device();
	generic_depth_device_data *const device_data = device->get_private_data<generic_depth_device_data>();
	const auto provided = describe_streamline_source(runtime);
	const bool game_provided = provided.selected;
	const char *provider_name = provided.provider == sunshine_scene_depth::provider_kind::ngx ? "NGX" : "Streamline";
	bool generic_current = false;
	{
		const std::shared_lock<std::shared_mutex> status_lock(s_mutex);
		generic_current = !provided.selected && device_data && (data.capture_ready || data.held) && data.selected_shader_resource != 0 &&
			data.selected_depth_stencil != 0 && data.native_access_present == device_data->native_present_index;
	}
	if (!troubleshooting)
	{
		if (game_provided)
		{
			// Ownership survives capture gaps. The dimensions describe the last valid
			// source, not a claim that the current presentation has usable depth.
			if (provided.last_valid.resource)
				ImGui::TextWrapped("Depth: %s | %ux%u | game provided", provider_name,
					provided.last_valid.width, provided.last_valid.height);
			else
				ImGui::TextWrapped("Depth: %s | waiting for first depth", provider_name);
			ImGui::SetItemTooltip("The game-provided source remains selected through capture gaps. Dimensions are from its last valid capture. Current availability and depth freshness are shown in the status rows.");
		}
		else if (data.selected_shader_resource != 0 && data.selected_depth_stencil != 0)
			ImGui::TextWrapped("Depth: Generic | %ux%u | %s",
				data.selected_desc.texture.width, data.selected_desc.texture.height,
				format_to_string(data.selected_desc.texture.format));
		else
			ImGui::TextWrapped("Depth: Generic | waiting for a usable buffer");
		if (!game_provided && generic_current)
		{
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.75f, 0.25f, 1.0f));
			ImGui::TextWrapped("Generic depth fallback is in use. Game-provided depth and camera data are not used.");
			ImGui::PopStyleColor();
		}

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
				ImGui::TextWrapped("Fresh: %.1f%% | Previous: %.1f%% | Unavailable: %.1f%%",
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
		return;
	}
	sunshine_game3d::draw_diagnostics(runtime);
}

static void draw_settings_panel(effect_runtime *runtime, int selected_tab = -1)
{
	if (!sunshine_game3d::draw(runtime))
		return;
	// All live prose is below the controls and tab bar. Tabs keep source and
	// preview controls stationary even when status messages wrap or disappear.
	if (!ImGui::BeginTabBar("Game3DDetails"))
		return;
	if (ImGui::BeginTabItem("Status", nullptr, selected_tab == 0 ? ImGuiTabItemFlags_SetSelected : 0))
	{
		if (ImGui::BeginChild("StatusContent"))
		{
			sunshine_game3d::draw_status(runtime);
			draw_depth_panel(runtime, false);
		}
		ImGui::EndChild();
		ImGui::EndTabItem();
	}
	if (ImGui::BeginTabItem("Troubleshooting", nullptr, selected_tab == 1 ? ImGuiTabItemFlags_SetSelected : 0))
	{
		if (ImGui::BeginChild("TroubleshootingContent", ImVec2(0, 0), 0, ImGuiWindowFlags_AlwaysVerticalScrollbar))
			draw_depth_panel(runtime, true);
		ImGui::EndChild();
		ImGui::EndTabItem();
	}
	if (ImGui::BeginTabItem("Calibration", nullptr, selected_tab == 2 ? ImGuiTabItemFlags_SetSelected : 0))
	{
		if (ImGui::BeginChild("CalibrationContent"))
			sunshine_game3d::draw_calibration(runtime);
		ImGui::EndChild();
		ImGui::EndTabItem();
	}
	ImGui::EndTabBar();
}

static void draw_settings_overlay(effect_runtime *runtime)
{
	draw_settings_panel(runtime);
}

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
void sunshine_game3d_test_draw_production_panel(effect_runtime *runtime, unsigned mode)
{
	// Exercise the actual production tabs without changing capture or settings.
	draw_settings_panel(runtime, static_cast<int>(mode) - 1);
}
#endif

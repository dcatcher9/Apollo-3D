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

// The linkage of depth_addon.cpp's shared state and depth-resource types
// (generic_depth_state.h), and in the test add-on the internals its exports
// (test_depth_addon_exports.cpp) reach. Only those two files include this
// header.
//
// The shipping add-on keeps every one of them internal (static) to
// depth_addon.cpp. The test add-on (SUNSHINE_SBS_RUNTIME_TEST_ADDON) makes the
// state's definitions inline and the functions declared below external, so
// that its exports can run in their own translation unit of the same DLL.
#pragma once

#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
#define SUNSHINE_DEPTH_ADDON_SHARED inline
#define SUNSHINE_DEPTH_ADDON_INTERNAL
#else
#define SUNSHINE_DEPTH_ADDON_SHARED static
#define SUNSHINE_DEPTH_ADDON_INTERNAL static
#endif

#include "generic_depth_state.h"

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
void on_bind_depth_stencil_impl(command_list *cmd_list, resource_view depth_stencil_view, bool direct_binding);
bool on_draw(command_list *cmd_list, uint32_t vertices, uint32_t instances, uint32_t, uint32_t);
bool on_draw_indexed(command_list *cmd_list, uint32_t indices, uint32_t instances, uint32_t, int32_t, uint32_t);
bool on_draw_indirect(command_list *cmd_list, indirect_command type, resource, uint64_t, uint32_t draw_count, uint32_t);
bool on_dispatch_mesh(command_list *cmd_list, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
bool on_copy_resource(command_list *cmd_list, resource source, resource dest);
bool on_copy_texture_region(command_list *cmd_list, resource source, uint32_t source_subresource, const subresource_box *source_box,
	resource dest, uint32_t subresource, const subresource_box *dest_box, filter_mode);
bool on_copy_buffer_to_texture(command_list *cmd_list, resource, uint64_t, uint32_t, uint32_t,
	resource dest, uint32_t subresource, const subresource_box *dest_box);
bool on_resolve_texture_region(command_list *cmd_list, resource source, uint32_t source_subresource, const subresource_box *source_box,
	resource dest, uint32_t subresource, uint32_t dest_x, uint32_t dest_y, uint32_t dest_z, format);
void resolve_transfer_lineage(state_tracking &state, const generic_depth_device_data &device_data);
hold_gate judge_hold_gate(const depth_selection_snapshot &snapshot, const sunshine_depth_capture::capture_set &captures,
	uint64_t held, uint64_t frame, uint32_t color_width, uint32_t color_height, bool resolve_depth, uint64_t grace);
sunshine_streamline::provider::source_status describe_streamline_source(effect_runtime *runtime);
// A fixture's call, made once at the next native depth frame's entry
// (sunshine_depth::begin_native_frame), between the UI's and depth's reads of
// the FG mode in that Present (SunshineGame3DTestOptionsAtDepthFrame).
inline std::atomic<void (*)(void *)> options_at_depth_frame{};
inline std::atomic<void *> options_at_depth_frame_context{};
#endif

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

// Every ReShade-to-capture lifecycle forward of Generic depth: what the
// capture owner (streamline_depth_capture.h) learns from ReShade's device,
// queue, command list and resource events. Only depth_addon.cpp includes it;
// its ReShade handlers (and those in generic_recording.inl) call these at the
// points where the calls were made before, in the same order relative to the
// Generic work of the same handler. No ReShade event is registered here.
//
// list_event, observe_command, observe_queue and retire_queue check for
// D3D12 themselves. The other forwards are made only where the handler has
// already decided (a D3D12 device, a sampled plane, a resource with an API
// identity); that check stays in the handler.
#pragma once

#include <cstddef>
#include <cstdint>
#include <d3d12.h>
#include <reshade.hpp>
#include "streamline_depth_capture.h"

namespace capture_inputs
{
	// The capture owner releases a D3D12 device's sources and snapshots (see
	// on_destroy_device); out receives its log line, if any.
	static inline bool release_device(reshade::api::device *device, char *out, std::size_t size)
	{
		return sunshine_streamline::depth_capture::release_device(device->get_native(), out, size);
	}

	// ReShade's own lifecycle for a D3D12 list; see depth_capture::observe_list_event.
	static void list_event(reshade::api::command_list *cmd_list, sunshine_streamline::depth_capture::list_event event)
	{
		if (cmd_list->get_device()->get_api() == reshade::api::device_api::d3d12)
			sunshine_streamline::depth_capture::observe_list_event(cmd_list->get_native(), event);
	}

	static inline void observe_command(reshade::api::command_list *cmd_list)
	{
		if (cmd_list->get_device()->get_api() == reshade::api::device_api::d3d12)
			sunshine_streamline::depth_capture::observe_command(cmd_list->get_native());
	}

	static inline void observe_queue(reshade::api::command_queue *cmd_queue)
	{
		if (cmd_queue->get_device()->get_api() == reshade::api::device_api::d3d12)
			sunshine_streamline::depth_capture::observe_queue(cmd_queue->get_native());
	}

	static inline void retire_queue(reshade::api::command_queue *cmd_queue)
	{
		if (cmd_queue->get_device()->get_api() == reshade::api::device_api::d3d12)
			sunshine_streamline::depth_capture::retire_queue(cmd_queue->get_native());
	}

	// ReShade reports a D3D12 creation state as its usage: the native state's
	// bits, with COMMON as general. undefined (an UNDEFINED initial layout) and
	// any other value with general's bit name no legacy state.
	static uint32_t creation_state(reshade::api::resource_usage usage)
	{
		if (usage == reshade::api::resource_usage::general) return D3D12_RESOURCE_STATE_COMMON;
		const auto value = static_cast<uint32_t>(usage);
		return usage == reshade::api::resource_usage::undefined || (value & static_cast<uint32_t>(reshade::api::resource_usage::general)) ? UINT32_MAX : value;
	}

	// A D3D12 depth-stencil resource was created in this ReShade creation state.
	static inline void resource_created(reshade::api::resource resource, reshade::api::resource_usage initial_state)
	{
		sunshine_streamline::depth_capture::resource_created(resource.handle, creation_state(initial_state));
	}

	// A tracked depth-stencil resource with an API identity was destroyed.
	static inline void resource_destroyed(uint64_t identity)
	{
		sunshine_streamline::depth_capture::resource_destroyed(identity);
	}

	// A D3D12 depth clear of a depth-stencil's subresource 0.
	static inline void observe_depth_clear(reshade::api::command_list *cmd_list, reshade::api::resource depth_stencil)
	{
		sunshine_streamline::depth_capture::observe_depth_clear(cmd_list->get_native(), depth_stencil.handle);
	}

	// The Present boundary of a D3D12 queue.
	static inline void present(reshade::api::command_queue *presenting_queue)
	{
		sunshine_streamline::depth_capture::present(presenting_queue->get_native());
	}
}

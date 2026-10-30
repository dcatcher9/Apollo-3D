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

// Generic depth's FrameActivityTrace route record, a statement of
// select_captured_depth (generic_capture.inl), included at its place inside
// that function, after the discovery hint and before the hold selection. It
// reads the function's locals (anchor, group, decision, observations,
// snapshot) and fills data.trace_capture only when FrameActivityTrace is on.
#pragma once

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

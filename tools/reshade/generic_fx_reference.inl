// SPDX-License-Identifier: GPL-3.0-only
// Static FX-reference helpers, same TU as depth_addon.cpp. Counterpart:
// exporter/fx_reference.inl. Caller guards, view/lock/resource work stay in place.
namespace sunshine_depth {
  static void begin_fx_reference_query(generic_depth_data *data, bool request_calibration) {
		// This is the current renderer's request, not a latch until effect reload.
		// Passive observation releases an earlier independent-renderer request
		// without changing scene selection, histogram evidence or capture state.
		data->native_depth_requested = request_calibration;
		data->native_ui_state = request_calibration ? native_depth_ui_state::waiting_depth : native_depth_ui_state::inactive;
		data->native_ui_samples = 0;
		data->native_ui_present = request_calibration ? data->native_access_present : 0;
  }

  static void resolve_fx_reference_calibration(generic_depth_data *data, frame_depth &output, depth_orientation orientation_override) {
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
  }

  static void finish_fx_reference_query(generic_depth_data *data, const frame_depth &output) {
		data->native_ui_samples = output.calibration_samples;
		data->native_ui_state = output.calibrated ? native_depth_ui_state::ready :
			(output.calibration_samples >= 3 && output.orientation == depth_orientation::automatic ?
				native_depth_ui_state::waiting_orientation : native_depth_ui_state::calibrating);

  }
}

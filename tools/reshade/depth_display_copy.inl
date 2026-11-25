// SPDX-License-Identifier: GPL-3.0-only
// Included at the unchanged copy block inside provider::begin.
    state::display_entry *target{};
    if (update.action == sunshine_game3d::depth_output::display_action::copy_fresh &&
        (display = [&] { const sunshine_game3d::slow_step step("depth display preparation");
          return prepare_display(runtime, *data, data->frame, present, target); }()) == display_status::ready) {
      const auto &packet = data->frame;
      copied = [&] { const sunshine_game3d::slow_step step("depth display copy");
#ifdef SUNSHINE_SBS_RUNTIME_TEST_ADDON
        // A test's refused copy records nothing, as a consumer refusal does.
        if (refuse_next_display_copy.exchange(false, std::memory_order_acq_rel)) return false;
#endif
        // The shared capture owner publishes the selected source through its read lease, in
        // the state the stereo passes sample. Neither renderer nor calibration chooses a
        // capture implementation or issues a texture transition.
        captured = {};
        constexpr std::uint32_t sampled_state = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        return runtime && commands && target->texture.handle && sunshine_streamline::depth_capture::copy_current(
          commands->get_native(), packet, target->texture.handle, sampled_state, &consumer,
          commands == runtime->get_command_queue()->get_immediate_command_list()); }();
      if (!copied) display = display_status::waiting_capture;
      // Its entry is the display from this copy on (prepare_display).
      else activate_display(*data, *target);
    }

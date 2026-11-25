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
        return sunshine_depth::copy_selected_depth(runtime, commands, packet, target->texture, captured, &consumer); }();
      if (!copied) display = display_status::waiting_capture;
      // Its entry is the display from this copy on (prepare_display).
      else activate_display(*data, *target);
    }

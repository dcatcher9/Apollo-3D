// SPDX-License-Identifier: GPL-3.0-only
// Included at the unchanged status-publication block inside provider::begin.
    const bool handoff_changed = display != data->last_display || consumer.result != data->last_consumer ||
      capture_info.selection != data->last_selection ||
      consumer.invalidation != data->last_loss || data->output.ready != data->last_output_ready;
    if (data->logging.due(sunshine_game3d::clock::game_now(), status != data->last_status ||
        capture_info.failure != data->last_capture_failure || handoff_changed, !previously_owned, !available)) {
      char text[2560]{};
      if (available) {
        selected_line line;
        line.path = path_name; line.status = status; line.output_ready = data->output.ready;
        line.output_projection_supplied = data->output.projection.supplied; line.display = display;
        line.present = present; line.command = commands->get_native();
        format_selected(text, sizeof(text), line, data->frame, consumer, capture_info);
      } else {
        // A failed acquisition has no output packet. Its zeroed fields say
        // nothing about the resource/camera that the provider actually tagged.
        format_held(text, sizeof(text), path_name, status, data->output.ready, display, capture_info,
          native_observer::counts());
      }
      append_selection(text, sizeof(text), capture_info, update, prior_depth_age, data->output.provided.jitter);
      {
        // The newest observation loss explains a revoked or missing capture;
        // an absent journal entry is not evidence that nothing was lost.
        const auto revision = sunshine_streamline::depth_observation_revision();
        sunshine_streamline::loss_diagnostics::event loss{};
        const bool known = revision && sunshine_streamline::query_depth_observation_loss(revision, loss);
        const auto since = revision >= data->logged_loss_revision ? revision - data->logged_loss_revision : 0;
        data->logged_loss_revision = revision;
        append_loss(text, sizeof(text), known, loss, known ? sunshine_game3d::clock::game_now() - loss.tick : 0, since);
      }
      sunshine_log::message(reshade::log::level::info, text);
      data->last_status = status;
      data->last_selection = capture_info.selection;
      data->last_capture_failure = capture_info.failure;
      data->last_display = display;
      data->last_consumer = consumer.result;
      data->last_loss = consumer.invalidation;
      data->last_output_ready = data->output.ready;
    }

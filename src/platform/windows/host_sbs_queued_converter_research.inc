// Included inside platf::dxgi after the converter class is defined.

int d3d_base_encode_device::convert_with_queued_research(
  platf::img_t &img,
  ::video::detail::optional_frame_time_point_t next_encode_target,
  rgb_output_t *rgb
) {
  // The private entry owns both tickets through recording, even if the converter retires
  // its research session. Keep their construction and destruction out of the shared frame.
  queued_converter_frame_t frame;
  return convert_frame(img, next_encode_target, rgb, &frame);
}

bool d3d_base_encode_device::queue_research_observation(
  queued_converter_frame_t &frame,
  matched_frame_slot_t &source,
  const models::depth_observation_views_t &previous,
  const queued_capture_t &capture,
  std::uint64_t observation_timestamp_us
) {
  if (probe_observer.record) {
    actual_converter_probe_v1::event_t event;
    event.kind = actual_converter_probe_v1::event_e::queue_begin;
    emit_converter_probe(std::move(event), &source, &previous, nullptr, source.srv.get(), capture.current_input, capture.frame_id);
  }

  std::string queue_error;
  auto ticket = depth_estimator->queue_pending_same_source_gpu_continuation_for_evaluation(
    source.analysis_input_srv(), capture.frame_id, observation_timestamp_us, queue_error
  );
  // Even an unavailable ticket can follow a rewrite of the borrowed estimator fields.
  latest_v2_lineage.reset();
  if (!ticket || !queued_ticket_matches(*ticket, source, capture.frame_id, observation_timestamp_us, capture.transfer)) {
    BOOST_LOG(error) << "Host SBS queued continuation unavailable; current flat identity: "sv << queue_error;
    return false;
  }

  frame.ticket = std::move(ticket);
  frame.source_slot = &source;
  const auto &draw = *frame.ticket;
  if (auto *pending = queued_research->pending_colour.get()) {
    const auto material = observe_pending_colour_material(
      *pending, draw, capture.desc, capture.transfer, capture.source, capture.content, capture.damage, capture.cursor
    );
    const auto operands = depth_estimator->pending_ocr_conditioner_working_views_for_evaluation(
      draw.ticket_sequence, draw.submitted_frame_id, draw.observation_timestamp_us
    );
    std::string archive_error;
    pending->owner_unavailable = !operands || !material.usable() ||
                                 !pending->lane.capture_pending(device_ctx.get(), *operands, draw, material, archive_error);
    if (pending->owner_unavailable) {
      pending->lane.release_geometry_lease();
      BOOST_LOG(warning) << "Pending current-colour owner unavailable: " << archive_error;
    }
  }
  if (probe_observer.record) {
    actual_converter_probe_v1::event_t event;
    event.kind = actual_converter_probe_v1::event_e::queued;
    event.queued = true;
    emit_converter_probe(std::move(event), &source, nullptr, &draw, draw.submitted_source.Get(), capture.current_input, capture.frame_id);
  }
  return true;
}

d3d_base_encode_device::queued_draw_t d3d_base_encode_device::select_queued_research_draw(
  queued_converter_frame_t &frame,
  const matched_frame_slot_t *candidate,
  const queued_capture_t &capture,
  std::chrono::steady_clock::time_point now,
  bool force_flat,
  bool snapshot_debug_inputs,
  bool producer_terminal,
  bool has_current_capture_mutex,
  bool current_interactive_move_size
) {
  queued_draw_t draw;
  draw.force_flat = force_flat;
  if (!queued_research) {
    // A retired session cannot authorize the ticket that the request still owns.
    draw.force_flat |= frame.ticket.has_value();
    return draw;
  }

  const bool warp_available = !snapshot_debug_inputs && !producer_terminal &&
                              host_sbs_renderer == models::host_sbs_renderer_e::parallax_v2;
  auto *pending = queued_research->pending_colour.get();
  if (pending) {
    const bool current_draw_allowed = has_current_capture_mutex && warp_available &&
                                      !current_interactive_move_size && !depth_authority_reprocess_pending &&
                                      !subtitle_context_reprocess_pending;
    draw.force_flat |= select_pending_colour_draw(
      *pending, frame, capture.current_input, current_draw_allowed, capture.desc, capture.transfer,
      now, capture.source, capture.content, capture.damage, capture.cursor, capture.frame_id
    );
  }

  // Recheck the exact pending source after queueing. The ticket's COM references alone
  // cannot establish which capture the slot now owns.
  auto *source = frame.source_slot;
  draw.same_source = frame.ticket && source && !draw.force_flat && warp_available &&
                     source == pending_matched_frame && source == candidate && source->frame_id == capture.frame_id &&
                     drawn_source_order.permits(source->frame_id) &&
                     matched_route_matches_current(
                       *source, capture.desc, capture.transfer, authority_generation(live_window_authority),
                       authority_generation(live_foreground_region), live_browser_authority_epoch, interactive_move_size_observed
                     );
  draw.force_flat |= frame.ticket.has_value() && !draw.same_source;
  if (frame.pending_colour_geometry) {
    draw.ticket = &*frame.pending_colour_geometry;
    draw.source = frame.pending_colour_source_slot;
    draw.current_colour = pending;
    draw.pixel_shader = pending->lane.pixel_shader();
  } else if (draw.same_source) {
    draw.ticket = &*frame.ticket;
    draw.source = source;
    draw.pixel_shader = queued_research->pixel_shader.get();
  }
  return draw;
}

ID3D11Buffer *d3d_base_encode_device::prepare_queued_draw_guard(const queued_ticket_t &ticket) {
  auto *buffer = queued_research->guard_constants.get();
  const auto guard = models::research_queued_render_guard_v1::make_expected(
    ticket.transaction_token, ticket.expected_work, ticket.observation_timestamp_us
  );
  device_ctx->UpdateSubresource(buffer, 0u, nullptr, &guard, 0u, 0u);
  return buffer;
}

bool d3d_base_encode_device::record_pending_colour_draw(
  const host_sbs_v2_draw_command_t &draw,
  pending_colour_research_t &pending
) {
  auto &lane = pending.lane;
  models::research_pending_current_colour_v1::scoped_negative_t negative;
  return negative.bind(device_ctx.get(), lane.negative_expected(), lane.negative_receipt(),
                       lane.current_constants().Get(), lane.current_snapshot().receipt.Get()) &&
         record_host_sbs_v2_draw(device_ctx.get(), draw);
}

bool d3d_base_encode_device::record_rejected_queued_draw_as_flat(
  host_sbs_v2_draw_command_t &draw,
  ID3D11ShaderResourceView *current_input
) {
  // A guarded draw can reject predication before recording commands. Submit current
  // identity unconditionally and restore the original context state on return.
  Microsoft::WRL::ComPtr<ID3D11Predicate> predicate;
  BOOL predicate_value = FALSE;
  device_ctx->GetPredication(&predicate, &predicate_value);
  const auto restore_predication = util::fail_guard([&] {
    device_ctx->SetPredication(predicate.Get(), predicate_value);
  });
  device_ctx->SetPredication(nullptr, FALSE);
  if (!update_sbs_constant_buffer(sbs_content_scale_x, sbs_content_scale_y, nullptr, nullptr)) {
    return false;
  }
  draw.pixel_shader = sbs_flat_identity_ps.get();
  draw.shader_resources = {current_input};
  draw.geometry_constants = sbs_reprojection_cbuffer.get();
  draw.subtitle_constants = nullptr;
  draw.queued_guard_constants = nullptr;
  draw.queued_guard_transaction = nullptr;
  draw.queued_guard_override = false;
  draw.shared_pin.reset();
  return record_host_sbs_v2_draw(device_ctx.get(), draw);
}

bool d3d_base_encode_device::initialize_queued_renderer_for_instance() {
  if (!probe_context || !depth_estimator) {
    return false;
  }
  if (queued_research) {
    return true;
  }

  namespace cache = models::host_sbs_shader_cache;
  const auto sources = cache::snapshot_sources(SUNSHINE_SHADERS_DIR, cache::parallax_v2_queued_renderer_specs);
  const bool exact_source = sources &&
                            cache::source_closure_sha256(sources) == cache::parallax_v2_queued_renderer_source_closure_sha256;
  const auto code = exact_source ? cache::get(sources, cache::parallax_v2_queued_renderer) : cache::bytecode_t {};

  auto research = std::make_unique<queued_converter_research_t>();
  const models::research_queued_render_guard_v1::expected_t disabled {};
  D3D11_BUFFER_DESC desc {};
  desc.ByteWidth = sizeof(disabled);
  desc.Usage = D3D11_USAGE_DEFAULT;
  desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
  const D3D11_SUBRESOURCE_DATA initial {&disabled, 0u, 0u};
  if (
    !code ||
    FAILED(device->CreatePixelShader(code->data(), code->size(), nullptr, &research->pixel_shader)) ||
    !research->pixel_shader ||
    FAILED(device->CreateBuffer(&desc, &initial, &research->guard_constants)) ||
    !research->guard_constants ||
    !depth_estimator->set_queued_same_source_gpu_continuation_research_enabled(true)
  ) {
    BOOST_LOG(error) << "Host SBS queued LIVE candidate unavailable; retaining legacy completion route."sv;
    return false;
  }
  queued_research = std::move(research);

  // The optional subtitle program cannot prevent the guarded queued Base draw.
  if (selected_matte_ready) {
    std::string error;
    if (!selected_shared_warp_programs.load_queued(device.get(), selected_shared_warp_root, error)) {
      BOOST_LOG(warning) << "Queued subtitle shared renderer unavailable; completed protection remains available: " << error;
    }
  }
  return true;
}

bool d3d_base_encode_device::initialize_pending_current_colour_for_instance(
  const models::research_pending_current_colour_v1::configuration_t &cfg,
  const models::host_sbs_gpu_completion_receipt::expected_t &genuine_warm,
  std::string &error
) {
  if (!cfg.enabled || !queued_research || queued_research->pending_colour || !depth_estimator) {
    return false;
  }
  const auto warm = depth_estimator->completed_ocr_conditioner_working_views_for_evaluation(genuine_warm);
  if (!warm) {
    error = "Genuine completed warm descriptors unavailable";
    return false;
  }

  static std::atomic<std::uint64_t> next_scope {1u};
  auto pending = std::make_unique<pending_colour_research_t>();
  pending->capture_epoch = next_scope.fetch_add(1u, std::memory_order_relaxed);
  const auto permission = next_scope.fetch_add(1u, std::memory_order_relaxed);
  if (!pending->capture_epoch || !permission || !pending->lane.prepare(device.get(), device_ctx.get(), *warm, cfg, pending->capture_epoch, permission, error)) {
    return false;
  }
  queued_research->pending_colour = std::move(pending);
  return true;
}

models::research_material_capture_v1::identity_t d3d_base_encode_device::observe_pending_colour_material(
  pending_colour_research_t &pending,
  const models::queued_same_source_gpu_continuation_research_views_t &owner,
  const D3D11_TEXTURE2D_DESC &desc,
  models::input_color_space transfer,
  const std::optional<std::chrono::steady_clock::time_point> &source,
  const std::optional<std::chrono::steady_clock::time_point> &content,
  const std::optional<detail::ddup_damage_snapshot_t> &damage,
  const platf::sbs_cursor::snapshot_t &cursor
) {
  namespace identity = models::research_material_capture_v1;
  const bool cursor_same = pending.previous_cursor && platf::sbs_cursor::same_presentation(*pending.previous_cursor, cursor);
  if (pending.last_identity.latest_capture_stamp && source == pending.last_identity.latest_capture_stamp && !cursor_same) {
    return {};
  }

  identity::input_t input;
  input.capture_epoch = pending.capture_epoch;
  input.source_stamp = source;
  input.content_stamp = content;
  auto &domain = input.domain;
  domain.width = desc.Width;
  domain.height = desc.Height;
  domain.format = desc.Format;
  domain.transfer = static_cast<std::uint32_t>(transfer);
  domain.mip_levels = desc.MipLevels;
  domain.array_size = desc.ArraySize;
  domain.samples = desc.SampleDesc.Count;
  domain.quality = desc.SampleDesc.Quality;
  domain.analysis_ltrb = {owner.input_region.left, owner.input_region.top, owner.input_region.right, owner.input_region.bottom};
  domain.content_ltrb = {owner.field_content.left, owner.field_content.top, owner.field_content.right, owner.field_content.bottom};
  domain.field_width = owner.field_width;
  domain.field_height = owner.field_height;
  domain.authority = static_cast<std::uint32_t>(owner.input_region.authority);
  domain.analysis_generation = owner.input_region.analysis_generation;
  domain.context_generation = owner.subtitle_context_generation;
  domain.root_authority_generation = authority_generation(live_window_authority);
  domain.region_authority_generation = authority_generation(live_foreground_region);
  domain.browser_epoch = live_browser_authority_epoch;

  const auto rect = damage_region(owner.input_region);
  const bool same_history = pending.previous_damage && damage && pending.previous_damage->history &&
                            pending.previous_damage->history.get() == damage->history.get();
  const bool roi_clean = same_history && rect &&
                         detail::query_ddup_damage_between(pending.previous_damage, damage, *rect) == detail::ddup_damage_intersection_e::unchanged;
  input.exact_analysis_pixels_unchanged = cursor_same &&
                                          (roi_clean || (same_history && content && pending.previous_content && content == pending.previous_content));
  // A DDup token identifies an acquired surface. It cannot prove there were no intervening
  // presentations because the snapshot does not carry AccumulatedFrames.
  input.complete_capture_interval_observed = false;
  const auto material = pending.material.observe(input);
  if (material.usable()) {
    pending.previous_damage = damage;
    pending.previous_content = content;
    pending.previous_cursor = cursor;
    pending.last_identity = material;
  }
  return material;
}

bool d3d_base_encode_device::select_pending_colour_draw(
  pending_colour_research_t &pending,
  queued_converter_frame_t &frame,
  ID3D11ShaderResourceView *current_input,
  bool current_draw_allowed,
  const D3D11_TEXTURE2D_DESC &live_source_desc,
  models::input_color_space input_color_space,
  std::chrono::steady_clock::time_point repeat_now,
  const std::optional<std::chrono::steady_clock::time_point> &current_source_timestamp,
  const std::optional<std::chrono::steady_clock::time_point> &current_content_timestamp,
  const std::optional<detail::ddup_damage_snapshot_t> &current_ddup_damage,
  const platf::sbs_cursor::snapshot_t &cursor,
  std::uint64_t frame_id
) {
  const auto same_source = [&](const matched_frame_slot_t &slot) {
    return slot.source_timestamp && current_source_timestamp && slot.source_timestamp == current_source_timestamp;
  };
  const bool force_flat =
    pending.owner_unavailable && pending_matched_frame && !same_source(*pending_matched_frame);

  const auto &owner = pending.lane.ticket();
  if (!owner) {
    return force_flat;
  }

  auto *owner_slot = pending_matched_frame;
  if (!owner_slot || owner_slot->frame_id != owner->submitted_frame_id) {
    // An ACK releases geometry ownership while the retirement bank and watermark survive.
    pending.lane.release_geometry_lease();
    return force_flat;
  }
  if (same_source(*owner_slot)) {
    return force_flat;
  }

  // Current colour and pending geometry have separate identities. The caller's permission
  // covers the current keyed input and the ordinary route and authority preconditions.
  if (
    !current_draw_allowed ||
    !subtitle_context.matches(owner->subtitle_context_generation) ||
    !owner_slot->subtitle_context_eligible ||
    owner->color_space != input_color_space ||
    !drawn_source_order.permits(frame_id) ||
    !models::host_sbs_matched_completion_is_current(false, repeat_now - owner_slot->captured_at) ||
    !matched_route_matches_current(
      *owner_slot,
      live_source_desc,
      input_color_space,
      authority_generation(live_window_authority),
      authority_generation(live_foreground_region),
      live_browser_authority_epoch,
      interactive_move_size_observed
    )
  ) {
    return true;
  }

  const auto material = observe_pending_colour_material(pending, *owner, live_source_desc, input_color_space, current_source_timestamp, current_content_timestamp, current_ddup_damage, cursor);
  const auto still_owned = depth_estimator->pending_ocr_conditioner_working_views_for_evaluation(owner->ticket_sequence, owner->submitted_frame_id, owner->observation_timestamp_us);
  std::string error;
  if (!still_owned || !material.usable() || !pending.lane.observe_current(device_ctx.get(), current_input, material, error)) {
    return true;
  }
  frame.pending_colour_geometry = *owner;
  frame.pending_colour_source_slot = owner_slot;
  return force_flat;
}

bool d3d_base_encode_device::queued_ticket_matches(
  const models::queued_same_source_gpu_continuation_research_views_t &ticket,
  const matched_frame_slot_t &slot,
  std::uint64_t frame_id,
  std::uint64_t observation_timestamp_us,
  models::input_color_space input_color_space
) const {
  // The slot owns its submission raster; the leased ticket owns its calibrated field mode.
  const auto expected_region = realize_depth_input_region(
    slot.depth_input_region,
    ticket.refined_live_geometry_active,
    static_cast<int>(ticket.field_width),
    static_cast<int>(ticket.field_height)
  );
  return ticket.postprocess_queued && ticket.both_unmap_calls_succeeded &&
         ticket.submitted_frame_id == frame_id && ticket.observation_timestamp_us == observation_timestamp_us &&
         ticket.submitted_source.Get() == slot.srv.get() && ticket.color_space == input_color_space &&
         ticket.postprocess_request_flags == 4u && ticket.expected_work == 1u && ticket.transaction_token != 0u &&
         ticket.subtitle_context_generation == slot.subtitle_context_generation &&
         expected_region && ticket.input_region == *expected_region &&
         ticket.field_content == expected_region->tensor_content && ticket.input_region.valid() &&
         ticket.field_width != 0u && ticket.field_height != 0u && ticket.field_content.valid() &&
         ticket.field_content.right <= ticket.field_width && ticket.field_content.bottom <= ticket.field_height &&
         ticket.base && ticket.state && ticket.transaction_raw &&
         queued_research && queued_research->guard_constants &&
         ticket.glyph_matte && ticket.display_band && ticket.condition_params && ticket.contour_identity;
}

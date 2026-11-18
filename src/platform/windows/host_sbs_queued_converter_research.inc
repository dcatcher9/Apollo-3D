// Included inside platf::dxgi after the converter class is defined.

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
  const bool force_flat = pending.owner_unavailable && std::ranges::any_of(matched_frame_slots, [&](const auto &slot) {
                            return slot.pending && !same_source(slot);
                          });

  const auto &owner = pending.lane.ticket();
  if (!owner) {
    return force_flat;
  }

  matched_frame_slot_t *owner_slot = nullptr;
  for (auto &candidate : matched_frame_slots) {
    if (candidate.pending && candidate.frame_id == owner->submitted_frame_id) {
      owner_slot = &candidate;
    }
  }
  if (!owner_slot) {
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
  models::input_color_space input_color_space,
  bool submitted_refined_geometry
) const {
  // The slot owns its submission raster. Its ticket must own the calibrated realized field.
  const auto expected_region = realize_depth_input_region(
    slot.depth_input_region,
    submitted_refined_geometry,
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

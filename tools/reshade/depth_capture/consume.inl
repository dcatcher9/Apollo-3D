// SPDX-License-Identifier: GPL-3.0-only
// Consumer admission, leases and copies; functions-only part of streamline_depth_capture.cpp.
// Types, namespace-scope state and test variables stay in the shell.

namespace sunshine_streamline::depth_capture {
  SUNSHINE_CAPTURE_INTERNAL {
    void describe_packet(const slot &value, std::uint64_t consumer, packet &out) {
      out = {};
      // Every consumer reads the copy-out once made (read_storage).
      const auto &storage = read_storage(value);
      const auto desc = value.nomination_only ? value.metadata.source->desc : storage->resource->GetDesc();
      out.metadata = value.metadata; out.ownership = storage;
      out.resource_id = value.metadata.source ? value.metadata.source->cookie : 0;
      if (value.nomination_only) out.device = reinterpret_cast<std::uint64_t>(value.metadata.source->device);
      else {
        out.texture = reinterpret_cast<std::uint64_t>(storage->resource.p);
        out.shader_resource = storage->shader_resource;
        out.device = reinterpret_cast<std::uint64_t>(storage->device);
      }
      out.queue = consumer; out.producer_queue = value.queue; out.capture_id = value.id; out.producer_fence = value.producer_fence;
      out.present = presented_by(value);
      out.width = static_cast<std::uint32_t>(desc.Width); out.height = desc.Height;
      out.area = sunshine_api_contract::normalized_extent(value.metadata.resource.area, out.width, out.height);
      out.format = typeless(desc.Format);
      // The snapshot's view was resolved at capture (depth_view); a
      // metadata-only nomination resolves its source's the same way.
      out.srv_format = value.nomination_only ? depth_view(desc.Format, value.metadata) : storage->view_format;
      out.pixel_ready = !value.nomination_only;
    }
  }

  void finish_diagnostic_texture(const diagnostic_ticket &ticket, bool successful) {
    if (!ticket) return;
    std::lock_guard lock(mutex);
    for (auto &entry : diagnostic_slots) if (entry.id == ticket.id && entry.texture == ticket.ownership) {
      if (entry.finished) return;
      entry.finished = true; entry.success = successful;
      if (!successful) invalidate(entry, capture_failure::evaluation_failed);
      return;
    }
  }

  static status acquire_auxiliary_texture(const diagnostic_ticket &ticket, diagnostic_texture &out,
      std::uint64_t consumer_queue, bool local) {
    out = {};
    const auto result = [&](status value) { out.result = value; return value; };
    if (!ticket) return result(status::unavailable);
    if (local && !consumer_queue) return result(status::unsupported_queue);
    std::unique_lock lock(mutex);
    // A queue already registered (observe_present, once per Present) needs no
    // second interface check, observation and lock round trip here.
    if (local) ensure_queue(lock, consumer_queue);
    const auto *consumer = local ? queue(consumer_queue) : nullptr;
    if (local && (!consumer || consumer->retiring)) return result(status::unsupported_queue);
    const auto native_consumer = consumer ? reinterpret_cast<std::uint64_t>(consumer->queue.p) : 0;
    for (auto &entry : diagnostic_slots) if (entry.id == ticket.id && entry.texture == ticket.ownership) {
      retire_dead_recordings(entry);
      out.capture_id = entry.id; out.producer_queue = entry.queue; out.producer_fence = entry.producer_fence;
      out.failure = entry.failure; out.producer_recording_retired = producer_recording_retired(entry);
      out.basis = entry.basis;
      const auto progress = local && entry.queue == native_consumer ? queue_progress{} : producer_progress(entry);
      out.producer_completed = progress.completed;
      const auto admission = local ? local_texture_admission(entry, native_consumer, consumer->device_identity, progress) :
        diagnostic_admission(entry, progress);
      if (admission != status::ready) return result(admission);
      const auto &texture = entry.texture;
      const auto desc = texture->resource->GetDesc();
      out.ownership = texture;
      out.texture = reinterpret_cast<std::uint64_t>(texture->resource.p);
      out.shared_handle = reinterpret_cast<std::uint64_t>(texture->shared_handle);
      out.device = reinterpret_cast<std::uint64_t>(texture->device);
      out.device_identity = texture->device_identity;
      out.resource_id = entry.metadata.source ? entry.metadata.source->cookie : 0;
      out.width = static_cast<std::uint32_t>(desc.Width); out.height = desc.Height; out.format = desc.Format;
      out.area = sunshine_api_contract::normalized_extent(entry.metadata.resource.area, out.width, out.height);
      return result(status::ready);
    }
    return result(status::unavailable);
  }

  status acquire_diagnostic_texture(const diagnostic_ticket &ticket, diagnostic_texture &out) {
    return acquire_auxiliary_texture(ticket, out, 0, false);
  }

  status acquire_local_texture(const diagnostic_ticket &ticket, std::uint64_t consumer_queue, diagnostic_texture &out) {
    return acquire_auxiliary_texture(ticket, out, consumer_queue, true);
  }

  void release_diagnostic_texture(const diagnostic_ticket &ticket) {
    if (!ticket) return;
    pooled_storage discard;
    std::lock_guard lock(mutex);
    for (auto &entry : diagnostic_slots) if (entry.id == ticket.id && entry.texture == ticket.ownership) {
      entry.diagnostic_released = true;
      // A canceled in-progress API call cannot become an accepted diagnostic.
      // Its command/fence obligations survive regardless of ticket ownership.
      if (!entry.finished) { entry.finished = true; entry.success = false; }
      break;
    }
    collect_diagnostics(discard);
  }

  // An immediate consumer is a ReShade runtime's immediate list, open outside
  // any render pass during its present/effects events. The runtime submits it
  // on the consumer queue and resets it afterwards, never replaying it. Once
  // registered (observe_runtime_list) it has a lifecycle like any game list.
  // Unregistered, its read is released at that list's next observed
  // submission, followed by the queue fence.
  static bool consume_owned(std::uint64_t native, const packet &value, std::uint64_t destination,
      std::uint32_t destination_state, consumer_diagnostic *diagnostic, bool auxiliary = false, bool local_auxiliary = false,
      bool immediate = false, local_view *lease = nullptr, std::uint32_t lease_format = 0) {
    if (diagnostic) *diagnostic = {};
    const auto result = [diagnostic](consumer_status why) {
      if (diagnostic) diagnostic->result = why;
      return why == consumer_status::ready;
    };
    // A snapshot released with its device (release_device) has no resource.
    if (!native || !value.ownership || value.ownership->released.load(std::memory_order_acquire))
      return result(consumer_status::missing_input);
    const auto submitted_list = native;
    // ReShade's registered immediate list was checked for this exact interface
    // at registration and observed by this Present's observe_present: skip
    // the repeated interface query and observation (and their lock).
    const bool registered = immediate && runtime_list(native);
    com_ptr<ID3D12GraphicsCommandList> checked;
    ID3D12GraphicsCommandList *list = nullptr;
    auto cookie = registered ? native_observer::get_recording_cookie(native) : 0;
    if (registered && cookie) list = reinterpret_cast<ID3D12GraphicsCommandList *>(native);
    else {
      if (!query_native(native, IID_ID3D12GraphicsCommandList, checked)) return result(consumer_status::unsupported_interface);
      native = reinterpret_cast<std::uint64_t>(checked.p);
      observe_command(native);
      cookie = native_observer::get_recording_cookie(native);
      list = checked.p;
    }
    com_ptr<ID3D12Device> device;
    if (list->GetType() != D3D12_COMMAND_LIST_TYPE_DIRECT) return result(consumer_status::unsupported_command_type);
    if (FAILED(list->GetDevice(__uuidof(ID3D12Device), reinterpret_cast<void **>(device.put())))) return result(consumer_status::missing_device);
    const auto device_identity = device_cookie(device.p);
    if (diagnostic) diagnostic->device_identity = device_identity;
    if (!device_identity) return result(consumer_status::missing_device_identity);
    // The snapshot's own identity: its device may be gone (release_device).
    const auto expected_device_identity = value.ownership->device_identity;
    if (diagnostic) diagnostic->expected_device_identity = expected_device_identity;
    if (device_identity != expected_device_identity) return result(consumer_status::device_mismatch);
    com_ptr<ID3D12Resource> target;
    source_ref retained_target;
    if (destination) {
      if (!query_native(destination, IID_ID3D12Resource, target) || target.p == value.ownership->resource.p ||
          !copy_state::supported(destination_state)) return result(consumer_status::invalid_destination);
      com_ptr<ID3D12Device> target_device;
      if (FAILED(target->GetDevice(IID_ID3D12Device, reinterpret_cast<void **>(target_device.put()))) ||
          device_cookie(target_device.p) != device_identity) return result(consumer_status::device_mismatch);
      const auto source_desc = value.ownership->resource->GetDesc(), target_desc = target->GetDesc();
      if (target_desc.Width != source_desc.Width || target_desc.Height != source_desc.Height ||
          (auxiliary ? !diagnostic_description(target_desc) ||
            auxiliary_copy_format(target_desc.Format) != auxiliary_copy_format(source_desc.Format) :
            !supported_description(target_desc) || typeless(target_desc.Format) != typeless(source_desc.Format)))
        return result(consumer_status::invalid_destination);
      if (auxiliary) {
        retained_target = retain_source(reinterpret_cast<std::uint64_t>(target.p));
        if (!retained_target) return result(consumer_status::invalid_destination);
      }
    }
    std::unique_lock lock(mutex);
    // A list in ReShade's lifecycle (or a registered runtime list) is read
    // under its recording lease, which also proves it is open and outside a
    // render pass. An unregistered runtime immediate list relies on its
    // runtime's contract and a submission lease.
    auto recording = command(native, cookie, true);
    const bool by_submission = lifecycle_view(recording ? &*recording : nullptr) != list_view::open;
    if (by_submission && !immediate) return result(consumer_status::observer_not_ready);
    // A lease needs the recording lease of a registered runtime list.
    if (lease && (by_submission || !registered)) return result(consumer_status::observer_not_ready);
    if (!by_submission) {
      if (diagnostic) { diagnostic->cookie = cookie; diagnostic->invalidation = recording->invalidation; }
      if (recording->invalid) return result(consumer_status::recording_invalid);
    } else recording = {};
    bool matching_id = false;
    auto *begin = auxiliary ? diagnostic_slots.data() : slots.data();
    auto *end = begin + (auxiliary ? diagnostic_slots.size() : slots.size());
    for (auto *current = begin; current != end; ++current) if (current->id == value.capture_id) {
      auto &entry = *current;
      matching_id = true;
      // The copy-out, once made, is read without the recording's Reset.
      const bool reads_copy_out = !auxiliary && entry.copied_out && entry.published && entry.published == value.ownership;
      if (entry.texture != value.ownership && !reads_copy_out) continue;
      if ((entry.consumer_queue || !auxiliary) && entry.consumer_queue != value.queue)
        return result(consumer_status::ownership_mismatch);
      retire_dead_recordings(entry);
      if (diagnostic) diagnostic->slot_invalid = entry.invalid;
      if (entry.invalid || !entry.finished || !entry.success) return result(consumer_status::capture_not_ready);
      if (auxiliary) {
        const auto *consumer = known_queue(value.queue);
        if (!consumer || consumer->retiring || consumer->device_identity != device_identity)
          return result(consumer_status::device_mismatch);
        const auto progress = local_auxiliary && entry.queue == value.queue ? queue_progress{} : producer_progress(entry);
        const auto admission = local_auxiliary ? local_texture_admission(entry, value.queue, device_identity, progress) :
          diagnostic_admission(entry, progress);
        if (admission != status::ready)
          return result(consumer_status::capture_not_ready);
        if (entry.auxiliary_destination && entry.auxiliary_destination != retained_target)
          return result(consumer_status::invalid_destination);
        // A snapshot leased on this recording sits in the shader-resource
        // state; a copy here would assume COMMON.
        if (entry.lease_list == native && entry.lease_cookie == cookie && (target.p || !lease))
          return result(consumer_status::ownership_mismatch);
        if (lease) {
          // Direct binding is same-queue only (no foreign or mixed producer),
          // reads the storage's own view, and never follows a copy.
          if (entry.queue != value.queue || !entry.texture->views.p) return result(consumer_status::capture_not_ready);
          const auto storage = entry.texture->resource->GetDesc().Format;
          const auto requested = static_cast<DXGI_FORMAT>(lease_format);
          if (!same_color_family(storage, requested) ||
              (entry.lease_list == native && entry.lease_cookie == cookie && entry.texture->view_format != requested))
            return result(consumer_status::invalid_destination);
        }
      }
      // One unsubmitted immediate list per slot; repeated reads share it.
      if (by_submission && entry.pending_immediate && entry.pending_immediate != submitted_list)
        return result(consumer_status::ownership_mismatch);
      for (unsigned i = 0; i != entry.consumers.size(); ++i) if (by_submission || !entry.consumers[i] || entry.consumers[i] == cookie) {
        if (!entry.preservation_only && entry.queue != value.queue) {
          // Recheck the actual slot before recording any read. A pending copy
          // must never add a queue dependency to the application's schedule.
          // A copy still in flight is read only on the runtime's immediate
          // list, which its runtime executes on value.queue at this Present,
          // once the game's own fences order that queue after it (U-C1).
          const auto *consumer = known_queue(value.queue);
          const auto progress = producer_progress(entry);
          sunshine_queue_order::verdict relied;
          if (!consumer || consumer->retiring || !(reads_copy_out || producer_recording_retired(entry)) ||
              !readable_elsewhere(entry, immediate ? value.queue : 0, progress, &relied))
            return result(consumer_status::capture_not_ready);
          if (!rely_on_queue_order(relied, entry, diagnostic)) return result(consumer_status::capture_not_ready);
        } else if (diagnostic) diagnostic->read_order = 1;
        if (by_submission) entry.pending_immediate = submitted_list;
        else { entry.consumers[i] = cookie; entry.consumer_recordings[i] = recording.life(); }
        entry.acquired = true;
        if (auxiliary) {
          entry.consumer_queue = value.queue;
          entry.auxiliary_destination = retained_target;
        }
        if (target.p) {
          // The registered consumer keeps the slot from reuse until its fence
          // passes, and the packet owns the storage: record without the lock.
          // All selected sources use this exact copy and consumer lease path.
          // Full subresource: depth plane or native auxiliary color pixels.
          ID3D12Resource *const source = value.ownership->resource.p;
          lock.unlock();
          native_observer::suppression_scope suppress;
          D3D12_RESOURCE_BARRIER transitions[2]{};
          transitions[0].Type = transitions[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
          transitions[0].Transition = {source, 0,
            auxiliary ? D3D12_RESOURCE_STATE_COMMON : value.ownership->rest, D3D12_RESOURCE_STATE_COPY_SOURCE};
          transitions[1].Transition = {target.p, 0, static_cast<D3D12_RESOURCE_STATES>(destination_state), D3D12_RESOURCE_STATE_COPY_DEST};
          list->ResourceBarrier(destination_state == D3D12_RESOURCE_STATE_COPY_DEST ? 1 : 2, transitions);
          D3D12_TEXTURE_COPY_LOCATION source_location{}, target_location{};
          source_location.pResource = source; source_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          target_location.pResource = target.p; target_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
          list->CopyTextureRegion(&target_location, 0, 0, 0, &source_location, nullptr);
          std::swap(transitions[0].Transition.StateBefore, transitions[0].Transition.StateAfter);
          std::swap(transitions[1].Transition.StateBefore, transitions[1].Transition.StateAfter);
          list->ResourceBarrier(destination_state == D3D12_RESOURCE_STATE_COPY_DEST ? 1 : 2, transitions);
        } else if (lease) {
          auto &texture = *entry.texture;
          const auto requested = static_cast<DXGI_FORMAT>(lease_format);
          if (texture.view_format != requested || !texture.shader_resource) {
            const auto handle = texture.views->GetCPUDescriptorHandleForHeapStart();
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = requested; view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING; view.Texture2D.MipLevels = 1;
            texture.device->CreateShaderResourceView(texture.resource.p, &view, handle);
            texture.shader_resource = handle.ptr;
            texture.view_format = requested;
          }
          if (entry.lease_list != native || entry.lease_cookie != cookie) {
            native_observer::suppression_scope suppress;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition = {texture.resource.p, 0, D3D12_RESOURCE_STATE_COMMON, sampled_state};
            list->ResourceBarrier(1, &barrier);
            entry.lease_list = native; entry.lease_cookie = cookie;
          }
          const auto desc = texture.resource->GetDesc();
          *lease = {};
          lease->view = texture.shader_resource;
          lease->resource = reinterpret_cast<std::uint64_t>(texture.resource.p);
          lease->capture_id = entry.id;
          lease->width = static_cast<std::uint32_t>(desc.Width); lease->height = desc.Height;
          lease->format = desc.Format; lease->view_format = requested;
        }
        return result(consumer_status::ready);
      }
      invalidate(entry, capture_failure::consumer_capacity); reason = status::exhausted; return result(consumer_status::consumer_capacity);
    }
    return result(matching_id ? consumer_status::ownership_mismatch : consumer_status::slot_missing);
  }

  bool mark_consumer(std::uint64_t command, const packet &value, consumer_diagnostic *diagnostic) {
    if (!value.pixel_ready) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = consumer_status::capture_not_ready; }
      return false;
    }
    return consume_owned(command, value, 0, 0, diagnostic);
  }

  bool copy_current(std::uint64_t command, const packet &value, std::uint64_t destination,
      std::uint32_t destination_state, consumer_diagnostic *diagnostic, bool immediate) {
    if (!destination || !value.pixel_ready) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = !destination ? consumer_status::invalid_destination : consumer_status::capture_not_ready; }
      return false;
    }
    return consume_owned(command, value, destination, destination_state, diagnostic, false, false, immediate);
  }

  static bool copy_auxiliary_texture(std::uint64_t command, std::uint64_t consumer_queue,
      const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
      consumer_diagnostic *diagnostic, bool local, bool immediate, local_view *lease = nullptr,
      std::uint32_t lease_format = 0) {
    const auto reject = [&](consumer_status value) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = value; }
      return false;
    };
    if (!command || !consumer_queue || !ticket) return reject(consumer_status::missing_input);
    if (!destination && !lease) return reject(consumer_status::invalid_destination);
    // A queue already registered (observe_present, once per Present) needs no
    // second interface check, observation and lock round trip here.
    packet selected;
    {
      std::unique_lock lock(mutex);
      ensure_queue(lock, consumer_queue);
      auto *owner = queue(consumer_queue);
      if (!owner || owner->retiring) return reject(consumer_status::missing_device);
      const auto found = std::find_if(diagnostic_slots.begin(), diagnostic_slots.end(),
        [&](const auto &entry) { return entry.id == ticket.id; });
      if (found == diagnostic_slots.end()) return reject(consumer_status::slot_missing);
      if (found->texture != ticket.ownership) return reject(consumer_status::ownership_mismatch);
      if (found->texture->device_identity != owner->device_identity) return reject(consumer_status::device_mismatch);
      const auto native_consumer = reinterpret_cast<std::uint64_t>(owner->queue.p);
      const auto progress = local && found->queue == native_consumer ? queue_progress{} : producer_progress(*found);
      const auto admission = local ? local_texture_admission(*found, native_consumer, owner->device_identity, progress) :
        diagnostic_admission(*found, progress);
      if (admission != status::ready)
        return reject(consumer_status::capture_not_ready);
      selected.capture_id = found->id;
      selected.ownership = ticket.ownership;
      selected.device = reinterpret_cast<std::uint64_t>(found->texture->device);
      selected.queue = reinterpret_cast<std::uint64_t>(owner->queue.p);
    }
    return consume_owned(command, selected, destination, destination_state, diagnostic, true, local, immediate, lease,
      lease_format);
  }

  bool copy_diagnostic_texture(std::uint64_t command, std::uint64_t consumer_queue,
      const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
      consumer_diagnostic *diagnostic, bool immediate) {
    return copy_auxiliary_texture(command, consumer_queue, ticket, destination, destination_state, diagnostic, false, immediate);
  }

  bool copy_local_texture(std::uint64_t command, std::uint64_t consumer_queue,
      const diagnostic_ticket &ticket, std::uint64_t destination, std::uint32_t destination_state,
      consumer_diagnostic *diagnostic, bool immediate) {
    return copy_auxiliary_texture(command, consumer_queue, ticket, destination, destination_state, diagnostic, true, immediate);
  }

  bool lease_local_view(std::uint64_t command, std::uint64_t consumer_queue, const diagnostic_ticket &ticket,
      std::uint32_t view_format, local_view &out, consumer_diagnostic *diagnostic) {
    out = {};
    if (!view_format) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = consumer_status::invalid_destination; }
      return false;
    }
    if (!runtime_list(command)) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = consumer_status::observer_not_ready; }
      return false;
    }
    local_view leased;
    if (!copy_auxiliary_texture(command, consumer_queue, ticket, 0, 0, diagnostic, true, true, &leased, view_format))
      return false;
    out = leased;
    return true;
  }

  void end_local_views(std::uint64_t command) {
    if (!command || !runtime_list(command)) return;
    const auto cookie = native_observer::get_recording_cookie(command);
    std::array<D3D12_RESOURCE_BARRIER, diagnostic_slot_limit> barriers{};
    UINT count = 0;
    std::lock_guard lock(mutex);
    for (auto &entry : diagnostic_slots) {
      if (!entry.lease_list || entry.lease_list != command) continue;
      // A submitted recording already ended: simultaneous-access storage
      // decayed to COMMON at its execution.
      if (entry.lease_cookie == cookie && cookie && entry.texture) {
        auto &barrier = barriers[count++];
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {entry.texture->resource.p, 0, sampled_state, D3D12_RESOURCE_STATE_COMMON};
      }
      entry.lease_list = entry.lease_cookie = 0;
    }
    if (!count) return;
    native_observer::suppression_scope suppress;
    reinterpret_cast<ID3D12GraphicsCommandList *>(command)->ResourceBarrier(count, barriers.data());
  }

  SUNSHINE_CAPTURE_INTERNAL {
    consumer_status preserved_admission(const slot &value, const preservation_ticket &ticket,
        std::uint64_t consumer, std::uint64_t consumer_device, std::uint64_t recording) {
      if (!value.id || value.id != ticket.id || !value.preservation_only || value.texture != ticket.ownership)
        return consumer_status::ownership_mismatch;
      if (value.invalid || !value.finished || !value.success ||
          (!value.producer_submitted && (!recording || value.command != recording)))
        return consumer_status::capture_not_ready;
      if (!value.metadata.source || value.metadata.source->device_identity != consumer_device)
        return consumer_status::device_mismatch;
      if (value.consumer_queue && value.consumer_queue != consumer) return consumer_status::ownership_mismatch;
      return consumer_status::ready;
    }
  }

  bool copy_preserved(std::uint64_t command, std::uint64_t consumer_queue, const preservation_ticket &ticket,
      std::uint64_t destination, std::uint32_t destination_state, consumer_diagnostic *diagnostic) {
    const auto reject = [&](consumer_status value) {
      if (diagnostic) { *diagnostic = {}; diagnostic->result = value; }
      return false;
    };
    if (!command || !consumer_queue || !ticket) return reject(consumer_status::missing_input);
    // ReShade's registered immediate list and its known queue were observed
    // by this Present's observe_present.
    if (!runtime_list(command) || !native_observer::get_recording_cookie(command)) observe_command(command);
    const auto cookie = native_observer::get_recording_cookie(command);
    packet selected;
    {
      std::unique_lock lock(mutex);
      ensure_queue(lock, consumer_queue);
      auto *owner = queue(consumer_queue);
      if (!owner || owner->retiring) return reject(consumer_status::missing_device);
      const auto native_queue = reinterpret_cast<std::uint64_t>(owner->queue.p);
      auto found = std::find_if(slots.begin(), slots.end(), [&](const auto &entry) { return entry.id == ticket.id; });
      if (found == slots.end()) return reject(consumer_status::slot_missing);
      const auto admission = preserved_admission(*found, ticket, native_queue, owner->device_identity, cookie);
      if (admission != consumer_status::ready) return reject(admission);
      found->consumer_queue = native_queue;
      describe_packet(*found, native_queue, selected);
    }
    return copy_current(command, selected, destination, destination_state, diagnostic);
  }
}

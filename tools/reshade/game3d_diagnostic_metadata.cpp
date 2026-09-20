// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_diagnostic_metadata.h"

#include "addon_lifetime.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <nlohmann/json.hpp>
#include <vector>

namespace sunshine_game3d {
  namespace {
    using namespace diagnostic;
    namespace sl = sunshine_streamline;
    using json = nlohmann::json;
    constexpr unsigned maximum_viewports = 4, maximum_tags = 32, maximum_features = 8, maximum_modules = 12;

    struct tag_value {
      stamp observation;
      tag_scope scope {};
      unsigned type {}, lifecycle {UINT32_MAX};
      sl::extent area;
      sl::abi_v2::resource descriptor;
      std::uint64_t tag_version {}, tag_extension {}, resource_pointer {};
      sl::guid tag_type {};
      bool tag_read {}, resource_read {}, full_descriptor {}, result_known {}, successful {}, version_one {};
    };

    struct constants_value {
      stamp observation;
      sl::common_constants common;
      std::array<std::uint8_t, 8> flags {};
      sl::decode_status decoded {};
      std::uint64_t version {}, extension {};
      bool present {}, readable {}, result_known {}, successful {}, version_one {};
      bool minimum_separation_available {};
      float minimum_separation {};
    };

    struct evaluation_value {
      stamp observation;
      unsigned feature {};
      bool present {}, successful {};
    };

    struct viewport_value {
      std::uint32_t viewport {};
      bool present {}, truncated {};
      constants_value constants;
      evaluation_value evaluation;
      std::array<tag_value, maximum_tags> tags {};
      unsigned count {};
    };

    struct module_value {
      std::uint64_t address {};
      char provider[16] {}, abi[64] {}, path[1024] {};
      std::array<unsigned, 4> file_version {}, product_version {};
      bool version_available {}, path_truncated {};
    };

    struct fg_value {
      stamp observation;
      unsigned mode {}, generated_frames {};
      bool present {}, supported {}, successful {};
    };

    struct storage {
      std::array<viewport_value, maximum_viewports> sl {};
      std::array<ngx_evaluation, maximum_features> ngx {};
      std::array<module_value, maximum_modules> modules {};
      std::array<fg_value, maximum_viewports> fg {};
      std::uint64_t session {}, armed_tick {};
      bool truncated {}, frame_observed {};
    };

    SRWLOCK lock = SRWLOCK_INIT;
    storage values;
    std::atomic<std::uint64_t> active {}, next_session {}, dropped {}, observed_session {};
    std::atomic<const resource_callbacks *> resource_observer {};

    bool same_guid(const sl::guid &a, const sl::guid &b) {
      return std::memcmp(&a, &b, sizeof(a)) == 0;
    }

    bool read_bytes(const void *source, void *destination, std::size_t bytes) noexcept {
      SIZE_T count {};
      return source && ReadProcessMemory(GetCurrentProcess(), source, destination, bytes, &count) && count == bytes;
    }

    void notify_resource(const resource_observation &value) noexcept {
      if (!value.observation.session || value.observation.session != diagnostic_metadata_generation()) return;
      const auto *observer = resource_observer.load(std::memory_order_acquire);
      if (observer && observer->resource) observer->resource(value);
    }

    void notify_finish(ui_resources::provider source, const stamp &at, std::uint64_t source_id, bool successful) noexcept {
      if (!at.session || at.session != diagnostic_metadata_generation()) return;
      const auto *observer = resource_observer.load(std::memory_order_acquire);
      if (observer && observer->finish) observer->finish(source, at, source_id, successful);
    }

    bool tag_layout_supported(const tag_value &value) noexcept {
      return value.version_one || (same_guid(value.tag_type, sl::tag_guid) && value.tag_version == 1);
    }

    void notify_tag(const tag_value &value) noexcept {
      const auto *catalog = ui_resources::find_sl(value.type);
      if (!catalog) return;
      resource_observation out;
      out.observation = value.observation;
      out.artifact_id = catalog->artifact_id;
      out.tag_type = value.type;
      out.lifecycle = value.lifecycle;
      out.area = value.area;
      out.version_one = value.version_one;
      out.readable = value.tag_read && value.resource_read;
      out.descriptor_supported = out.readable && tag_layout_supported(value) &&
                                 (!value.resource_pointer || value.descriptor.type == 0);
      if (out.descriptor_supported) {
        out.native = reinterpret_cast<std::uint64_t>(value.descriptor.native);
        out.native_state = value.descriptor.state;
        // SL's zero/unspecified state still needs independently observed proof.
        out.state_declared = out.native_state != 0 && out.native_state != UINT32_MAX;
      }
      notify_resource(out);
    }

    template<class F>
    void write(const stamp &at, F operation) noexcept {
      if (!at.session || at.session != diagnostic_metadata_generation()) {
        return;
      }
      if (!TryAcquireSRWLockExclusive(&lock)) {
        ++dropped;
        return;
      }
      if (at.session == active.load(std::memory_order_acquire) && at.session == values.session) {
        operation();
      }
      ReleaseSRWLockExclusive(&lock);
    }

    template<class F>
    void write_frame(const stamp &at, F operation) noexcept {
      write(at, [&] {
        operation();
        values.frame_observed = true;
        observed_session.store(at.session, std::memory_order_release);
      });
    }

    viewport_value *viewport(unsigned id) noexcept {
      for (auto &entry : values.sl) {
        if (entry.present && entry.viewport == id) {
          return &entry;
        }
      }
      for (auto &entry : values.sl) {
        if (!entry.present) {
          entry.present = true;
          entry.viewport = id;
          return &entry;
        }
      }
      values.truncated = true;
      return nullptr;
    }

    void record_tag(const tag_value &tag) noexcept {
      auto *entry = viewport(tag.observation.viewport);
      if (!entry) {
        return;
      }
      for (unsigned i = 0; i != entry->count; ++i) {
        if (entry->tags[i].type == tag.type) {
          if (entry->tags[i].observation.sequence <= tag.observation.sequence) {
            entry->tags[i] = tag;
          }
          return;
        }
      }
      if (entry->count == maximum_tags) {
        entry->truncated = true;
        values.truncated = true;
        return;
      }
      entry->tags[entry->count++] = tag;
    }

    tag_value copy_tag(const stamp &at, const sl::abi_v2::resource_tag &tag, tag_scope scope) noexcept {
      tag_value out;
      out.observation = at;
      out.scope = scope;
      out.tag_read = true;
      out.type = tag.type;
      out.lifecycle = tag.lifecycle;
      out.area = tag.area;
      out.tag_type = tag.base.type;
      out.tag_version = tag.base.version;
      out.tag_extension = reinterpret_cast<std::uint64_t>(tag.base.next);
      out.resource_pointer = reinterpret_cast<std::uint64_t>(tag.resource_ptr);
      // A null resource explicitly clears a tag. Preserve it as null, not as a
      // stale prior address. Unknown layouts expose their header only.
      if (!tag.resource_ptr) {
        out.resource_read = true;
        return out;
      }
      if (!same_guid(tag.base.type, sl::tag_guid) || tag.base.version != 1) {
        return out;
      }
      auto &resource = out.descriptor;
      constexpr auto prefix = offsetof(sl::abi_v2::resource, height) + sizeof(resource.height);
      const bool header_read = read_bytes(tag.resource_ptr, &resource.base, sizeof(resource.base));
      const sl::guid empty {};
      const bool known = header_read &&
                         ((same_guid(resource.base.type, sl::resource_guid) && resource.base.version == 1) ||
                          (same_guid(resource.base.type, empty) && resource.base.version == 0));
      if (known) {
        out.resource_read = read_bytes(tag.resource_ptr, &resource, prefix);
        sl::abi_v2::resource complete {};
        if (out.resource_read && read_bytes(tag.resource_ptr, &complete, sizeof(complete))) {
          resource = complete;
          out.full_descriptor = true;
        }
      }
      return out;
    }

    std::string address(std::uint64_t value) {
      char out[24] {};
      std::snprintf(out, sizeof(out), "0x%llx", static_cast<unsigned long long>(value));
      return out;
    }

    std::string guid_string(const sl::guid &value) {
      char out[40] {};
      std::snprintf(out, sizeof(out), "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", value.a, value.b, value.c, value.d[0], value.d[1], value.d[2], value.d[3], value.d[4], value.d[5], value.d[6], value.d[7]);
      return out;
    }

    json to_json(const stamp &value) {
      return {{"diagnostic_session", value.session}, {"adapter_observation_generation", value.epoch}, {"observation_sequence", value.sequence}, {"tick_ms", value.tick}, {"viewport", value.viewport == UINT32_MAX ? json(nullptr) : json(value.viewport)}, {"frame_token_identity", address(value.frame_token)}, {"frame_numeric", value.numeric_frame ? json(value.frame_numeric) : json(nullptr)}, {"command_identity", address(value.command)}};
    }

    json ngx_result(std::uint32_t value, bool attempted) {
      if (!attempted) return nullptr;
      const char *name = "unknown";
      if (value == 1) name = "Success";
      else if (value == 0xbad00005) name = "FAIL_InvalidParameter";
      else if (value == 0xbad00007) name = "FAIL_NotInitialized";
      else if (value == 0xbad00010) name = "FAIL_UnsupportedParameter";
      else if (value == 0xbad00012) name = "FAIL_NotImplemented";
      return {{"code", value}, {"hex", address(value)}, {"name", name}};
    }


    json matrix(const sl::matrix4 &value) {
      json result = json::array();
      for (const auto &row : value.m) {
        result.push_back(row);
      }
      return result;
    }

    json camera(const constants_value &value) {
      const auto &c = value.common;
      json result {{"observation", to_json(value.observation)}, {"readable", value.readable}, {"decode_status", unsigned(value.decoded)}, {"struct_version", value.version}, {"extension_identity", address(value.extension)}, {"result_known", value.result_known}, {"successful", value.result_known ? json(value.successful) : json(nullptr)}};
      if (!value.readable) {
        return result;
      }
      result["common"] = {{"camera_view_to_clip", matrix(c.camera_view_to_clip)}, {"clip_to_camera_view", matrix(c.clip_to_camera_view)}, {"clip_to_lens_clip", matrix(c.clip_to_lens_clip)}, {"clip_to_prev_clip", matrix(c.clip_to_prev_clip)}, {"prev_clip_to_clip", matrix(c.prev_clip_to_clip)}, {"jitter_offset", c.jitter_offset}, {"motion_vector_scale", c.motion_vector_scale}, {"camera_pinhole_offset", c.camera_pinhole_offset}, {"camera_position", c.camera_position}, {"camera_up", c.camera_up}, {"camera_right", c.camera_right}, {"camera_forward", c.camera_forward}, {"camera_near", c.camera_near}, {"camera_far", c.camera_far}, {"camera_fov", c.camera_fov}, {"camera_aspect", c.camera_aspect}, {"invalid_motion_vector", c.invalid_motion_vector}};
      result["flags"] = {{"depth_inverted", value.flags[0]}, {"camera_motion_included", value.flags[1]}, {"motion_vectors_3d", value.flags[2]}, {"reset", value.flags[3]}, {"orthographic_projection", value.flags[4]}, {"motion_vectors_dilated", value.flags[5]}, {"motion_vectors_jittered", value.flags[6]}, {"not_rendering_game_frames", value.version_one ? json(value.flags[7]) : json(nullptr)}};
      result["min_relative_linear_depth_object_separation"] = value.minimum_separation_available ? json(value.minimum_separation) : json(nullptr);
      return result;
    }

    json tag_json(const tag_value &value) {
      const auto &r = value.descriptor;
      return {{"observation", to_json(value.observation)}, {"scope", unsigned(value.scope)}, {"type", value.type}, {"lifecycle", value.version_one ? json(nullptr) : json(value.lifecycle)}, {"extent", {{"left", value.area.left}, {"top", value.area.top}, {"width", value.area.width}, {"height", value.area.height}}}, {"tag_readable", value.tag_read}, {"tag_struct_version", value.tag_version}, {"tag_structure_guid", guid_string(value.tag_type)}, {"tag_extension_identity", address(value.tag_extension)}, {"resource_argument_identity", address(value.resource_pointer)}, {"resource_readable", value.resource_read}, {"resource_type", r.type}, {"native_identity", address(reinterpret_cast<std::uint64_t>(r.native))}, {"memory_identity", address(reinterpret_cast<std::uint64_t>(r.memory))}, {"view_identity", address(reinterpret_cast<std::uint64_t>(r.view))}, {"state", r.state}, {"width", value.version_one ? json(nullptr) : json(r.width)}, {"height", value.version_one ? json(nullptr) : json(r.height)}, {"resource_struct_version", r.base.version}, {"resource_structure_guid", guid_string(r.base.type)}, {"resource_extension_identity", address(reinterpret_cast<std::uint64_t>(r.base.next))}, {"full_descriptor_readable", value.full_descriptor}, {"format", value.full_descriptor ? json(r.native_format) : json(nullptr)}, {"mip_levels", value.full_descriptor ? json(r.mip_levels) : json(nullptr)}, {"array_layers", value.full_descriptor ? json(r.array_layers) : json(nullptr)}, {"flags", value.full_descriptor ? json(r.flags) : json(nullptr)}, {"usage", value.full_descriptor ? json(r.usage) : json(nullptr)}, {"gpu_virtual_address_identity", value.full_descriptor ? json(address(r.gpu_virtual_address)) : json(nullptr)}, {"result_known", value.result_known}, {"successful", value.result_known ? json(value.successful) : json(nullptr)}};
    }

    const char *tag_availability(const tag_value &value) noexcept {
      if (!value.tag_read) return "unreadable";
      if (!tag_layout_supported(value)) return "unsupported";
      if (!value.resource_read) {
        return value.descriptor.base.version ? "unsupported" : "unreadable";
      }
      if (value.resource_pointer && value.descriptor.type != 0) return "unsupported";
      return value.descriptor.native ? "non_null" : "null";
    }

    const char *parameter_availability(const parameter &value) noexcept {
      if (!value.getter_available) return "getter_unavailable";
      if (!value.successful) return "query_failed";
      if (value.type != parameter_type::resource) return "unsupported";
      return value.integer ? "non_null" : "null";
    }

    json ui_inventory(const storage &snapshot) {
      json inventory = json::array();
      for (const auto &entry : ui_resources::catalog) {
        const bool is_sl = entry.source == ui_resources::provider::streamline;
        json row {{"artifact_id", entry.artifact_id}, {"provider", is_sl ? "streamline" : "ngx"},
          {"name", entry.name}, {"file_stem", entry.file_stem}, {"semantic", entry.semantic},
          {"role", ui_resources::role_name(entry.content)}, {"tag_type", is_sl ? json(entry.tag_type) : json(nullptr)},
          {"parameter_key", entry.parameter_key ? json(entry.parameter_key) : json(nullptr)},
          {"transfer_status", "unknown"}, {"runtime_support", "unknown"}, {"state", "unobserved"},
          {"capture_status", nullptr}, {"observations", json::array()}};
        std::uint64_t newest_tick {}, newest_sequence {};
        const auto append = [&](json item, const stamp &at, const char *availability) {
          item["state"] = availability;
          item["observation"] = to_json(at);
          if (row["observations"].empty() || at.tick > newest_tick || (at.tick == newest_tick && at.sequence >= newest_sequence)) {
            row["state"] = availability;
            newest_tick = at.tick;
            newest_sequence = at.sequence;
          }
          row["observations"].push_back(std::move(item));
        };
        if (is_sl) {
          for (const auto &view : snapshot.sl) {
            if (!view.present) continue;
            for (unsigned i = 0; i != view.count; ++i) {
              const auto &tag = view.tags[i];
              if (tag.type != entry.tag_type) continue;
              json item {{"native_identity", tag.resource_read ? json(address(reinterpret_cast<std::uint64_t>(tag.descriptor.native))) : json(nullptr)},
                {"scope", unsigned(tag.scope)}, {"active_rect", {{"left", tag.area.left}, {"top", tag.area.top}, {"width", tag.area.width}, {"height", tag.area.height}}},
                {"active_rect_source", "sl_tag_extent_zero_means_full_resource"},
                {"lifecycle", tag.version_one ? json(nullptr) : json(tag.lifecycle)},
                {"native_state", tag.resource_read ? json(tag.descriptor.state) : json(nullptr)},
                {"sdk_result_known", tag.result_known}, {"sdk_successful", tag.result_known ? json(tag.successful) : json(nullptr)}};
              append(std::move(item), tag.observation, tag_availability(tag));
            }
          }
        } else {
          for (const auto &feature : snapshot.ngx) {
            if (!feature.observation.session) continue;
            for (unsigned i = 0; i != std::min(feature.parameter_count, maximum_parameters); ++i) {
              const auto &p = feature.parameters[i];
              if (std::strcmp(p.name, entry.parameter_key)) continue;
              json item {{"native_identity", p.getter_available && p.successful ? json(address(p.integer)) : json(nullptr)},
                {"owner_identity", address(feature.owner)}, {"feature_handle_identity", address(feature.handle)}, {"source_id", feature.source_id},
                {"getter_available", p.getter_available}, {"getter_successful", p.getter_available ? json(p.successful) : json(nullptr)},
                {"getter_result", p.getter_available ? json(p.result) : json(nullptr)},
                {"getter_result_detail", ngx_result(p.result, p.getter_available)},
                {"active_rect", nullptr}, {"active_rect_source", "not_observed_for_this_parameter"},
                {"sdk_result_known", feature.result_known}, {"sdk_successful", feature.result_known ? json(feature.successful) : json(nullptr)}};
              append(std::move(item), feature.observation, parameter_availability(p));
            }
          }
        }
        inventory.push_back(std::move(row));
      }
      return inventory;
    }

    std::string unavailable_snapshot(const char *status) {
      json result {{"schema", "sunshine-game3d-metadata-1"}, {"status", status}, {"truncated", true},
        {"ui_resources", ui_inventory(storage {})}};
      for (auto &row : result["ui_resources"]) row["state"] = "snapshot_unavailable";
      return result.dump();
    }
  }  // namespace

  void arm_diagnostic_metadata(bool enabled) {
    if (!enabled) {
      active.store(0, std::memory_order_release);
      return;
    }
    if (diagnostic_metadata_generation()) {
      return;
    }
    AcquireSRWLockExclusive(&lock);
    if (!active.load(std::memory_order_relaxed)) {
      const auto modules = values.modules;
      const auto fg = values.fg;
      values = {};
      values.modules = modules;
      values.fg = fg;
      values.session = ++next_session;
      values.armed_tick = GetTickCount64();
      dropped = 0;
      observed_session.store(0, std::memory_order_relaxed);
      active.store(values.session, std::memory_order_release);
    }
    ReleaseSRWLockExclusive(&lock);
  }

  std::uint64_t diagnostic_metadata_generation() noexcept {
    return sunshine_addon_lifetime::stopping() ? 0 : active.load(std::memory_order_acquire);
  }

  bool has_diagnostic_frame_observation() noexcept {
    const auto session = diagnostic_metadata_generation();
    return session && observed_session.load(std::memory_order_acquire) == session;
  }

  namespace diagnostic {
    void set_resource_callbacks(const resource_callbacks *callbacks) noexcept {
      resource_observer.store(callbacks, std::memory_order_release);
    }
    void observe_module(HMODULE module, const char *provider, const char *abi) noexcept try {
      if (!module || sunshine_addon_lifetime::stopping()) {
        return;
      }
      module_value value;
      value.address = reinterpret_cast<std::uint64_t>(module);
      std::snprintf(value.provider, sizeof(value.provider), "%s", provider);
      std::snprintf(value.abi, sizeof(value.abi), "%s", abi);
      wchar_t path[32768] {};
      const auto length = GetModuleFileNameW(module, path, DWORD(std::size(path)));
      if (length && length < std::size(path)) {
        const int required = WideCharToMultiByte(CP_UTF8, 0, path, int(length), nullptr, 0, nullptr, nullptr);
        value.path_truncated = required >= int(sizeof(value.path));
        if (!value.path_truncated) {
          WideCharToMultiByte(CP_UTF8, 0, path, int(length), value.path, int(sizeof(value.path) - 1), nullptr, nullptr);
        }
        DWORD unused {};
        const auto size = GetFileVersionInfoSizeW(path, &unused);
        if (size && size <= 1024 * 1024) {
          std::vector<unsigned char> bytes(size);
          VS_FIXEDFILEINFO *info {};
          UINT info_size {};
          if (GetFileVersionInfoW(path, 0, size, bytes.data()) && VerQueryValueW(bytes.data(), L"\\", reinterpret_cast<void **>(&info), &info_size) && info_size >= sizeof(*info) && info->dwSignature == 0xfeef04bd) {
            value.version_available = true;
            value.file_version = {HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS), HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS)};
            value.product_version = {HIWORD(info->dwProductVersionMS), LOWORD(info->dwProductVersionMS), HIWORD(info->dwProductVersionLS), LOWORD(info->dwProductVersionLS)};
          }
        }
      } else {
        value.path_truncated = true;
      }
      AcquireSRWLockExclusive(&lock);
      auto *destination = &values.modules.back();
      for (auto &entry : values.modules) {
        if (!entry.address || entry.address == value.address) {
          destination = &entry;
          break;
        }
      }
      if (destination->address && destination->address != value.address) {
        values.truncated = true;
      }
      *destination = value;
      ReleaseSRWLockExclusive(&lock);
    } catch (...) {
      // Optional metadata cannot make source-hook discovery fail.
    }

    void observe_sl_constants(const stamp &at, const sl::abi_v1::constants &raw, bool readable, sl::decode_status decoded) noexcept {
      constants_value value;
      value.observation = at;
      value.common = raw.common;
      value.present = true;
      value.readable = readable;
      value.decoded = decoded;
      value.version_one = true;
      value.version = 1;
      value.extension = reinterpret_cast<std::uint64_t>(raw.ext);
      value.flags = {raw.depth_inverted, raw.camera_motion_included, raw.motion_vectors_3d, raw.reset, raw.orthographic_projection, raw.motion_vectors_dilated, raw.motion_vectors_jittered, raw.not_rendering_game_frames};
      write_frame(at, [&] {
        if (auto *entry = viewport(at.viewport); entry && entry->constants.observation.sequence <= at.sequence) {
          entry->constants = value;
        }
      });
    }

    void observe_sl_constants(const stamp &at, const sl::abi_v2::constants &raw, bool readable, sl::decode_status decoded, bool minimum_separation_available) noexcept {
      constants_value value;
      value.observation = at;
      value.common = raw.common;
      value.present = true;
      value.readable = readable;
      value.decoded = decoded;
      value.version = raw.base.version;
      value.extension = reinterpret_cast<std::uint64_t>(raw.base.next);
      value.minimum_separation_available = minimum_separation_available;
      value.minimum_separation = raw.min_relative_linear_depth_object_separation;
      value.flags = {raw.depth_inverted, raw.camera_motion_included, raw.motion_vectors_3d, raw.reset, raw.orthographic_projection, raw.motion_vectors_dilated, raw.motion_vectors_jittered, 0};
      write_frame(at, [&] {
        if (auto *entry = viewport(at.viewport); entry && entry->constants.observation.sequence <= at.sequence) {
          entry->constants = value;
        }
      });
    }

    void observe_sl_tag_v1(const stamp &at, const sl::abi_v1::resource *resource, unsigned type, const sl::extent *area) noexcept {
      if (!at.session) {
        return;
      }
      tag_value value;
      value.observation = at;
      value.type = type;
      value.version_one = true;
      value.resource_pointer = reinterpret_cast<std::uint64_t>(resource);
      sl::abi_v1::resource raw {};
      value.tag_read = !area || read_bytes(area, &value.area, sizeof(value.area));
      value.resource_read = !resource || read_bytes(resource, &raw, sizeof(raw));
      if (value.resource_read) {
        value.descriptor.type = raw.type;
        value.descriptor.native = raw.native;
        value.descriptor.memory = raw.memory;
        value.descriptor.view = raw.view;
        value.descriptor.state = raw.state;
        value.descriptor.base.next = raw.ext;
      }
      notify_tag(value);
      write_frame(at, [&] {
        record_tag(value);
      });
    }

    void observe_sl_tags_v2(const stamp &input, const sl::abi_v2::viewport &raw_viewport, const sl::abi_v2::resource_tag *tags, unsigned count, tag_scope scope) noexcept {
      if (!input.session) {
        return;
      }
      stamp at = input;
      sl::abi_v2::viewport copied {};
      if (read_bytes(&raw_viewport, &copied, sizeof(copied)) && same_guid(copied.base.type, sl::viewport_guid) && copied.base.version == 1 && !copied.base.next) {
        at.viewport = copied.value;
      } else {
        at.viewport = UINT32_MAX;
      }
      std::array<tag_value, maximum_tags> snapshot {};
      const unsigned size = std::min(count, maximum_tags);
      bool unreadable = false;
      for (unsigned i = 0; i != size; ++i) {
        sl::abi_v2::resource_tag tag {};
        if (!tags || !read_bytes(tags + i, &tag, sizeof(tag))) {
          unreadable = true;
          break;
        }
        snapshot[i] = copy_tag(at, tag, scope);
      }
      for (unsigned i = 0; i != size && snapshot[i].tag_read; ++i) notify_tag(snapshot[i]);
      write_frame(at, [&] {
        if (auto *entry = viewport(at.viewport); entry && (unreadable || count > maximum_tags)) {
          entry->truncated = true;
          values.truncated = true;
        }
        for (unsigned i = 0; i != size && snapshot[i].tag_read; ++i) {
          record_tag(snapshot[i]);
        }
      });
    }

    void observe_sl_inputs_v2(const stamp &input, const sl::base_structure **inputs, unsigned count) noexcept {
      if (!input.session) {
        return;
      }
      stamp at = input;
      std::array<tag_value, maximum_tags> tags {};
      unsigned total {}, visited {};
      bool truncated = count > maximum_tags;
      for (unsigned i = 0; i != std::min(count, maximum_tags); ++i) {
        const sl::base_structure *current {};
        if (!inputs || !read_bytes(inputs + i, &current, sizeof(current))) {
          truncated = true;
          break;
        }
        while (current && visited++ < 64) {
          sl::base_structure header {};
          if (!read_bytes(current, &header, sizeof(header))) {
            truncated = true;
            break;
          }
          if (same_guid(header.type, sl::viewport_guid) && header.version == 1) {
            sl::abi_v2::viewport view {};
            if (read_bytes(current, &view, sizeof(view))) {
              at.viewport = view.value;
            } else {
              truncated = true;
            }
          } else if (same_guid(header.type, sl::tag_guid)) {
            sl::abi_v2::resource_tag tag {};
            if (total < maximum_tags && read_bytes(current, &tag, sizeof(tag))) {
              tags[total++] = copy_tag(at, tag, tag_scope::evaluation);
            } else {
              truncated = true;
            }
          }
          current = static_cast<const sl::base_structure *>(header.next);
        }
        if (current) {
          truncated = true;
        }
      }
      for (unsigned i = 0; i != total; ++i) {
        tags[i].observation.viewport = at.viewport;
        notify_tag(tags[i]);
      }
      write_frame(at, [&] {
        if (auto *entry = viewport(at.viewport); entry && truncated) {
          entry->truncated = true;
          values.truncated = true;
        }
        for (unsigned i = 0; i != total; ++i) {
          tags[i].observation.viewport = at.viewport;
          record_tag(tags[i]);
        }
      });
    }

    void observe_sl_evaluation(const stamp &at, unsigned feature, bool successful) noexcept {
      write_frame(at, [&] {
        if (auto *entry = viewport(at.viewport); entry && entry->evaluation.observation.sequence <= at.sequence) {
          entry->evaluation = {at, feature, true, successful};
        }
      });
    }

    void finish_sl_call(const stamp &at, bool successful) noexcept {
      write(at, [&] {
        for (auto &entry : values.sl) {
          if (entry.present) {
            auto &c = entry.constants;
            if (c.present && c.observation.sequence == at.sequence && c.observation.epoch == at.epoch) {
              c.result_known = true;
              c.successful = successful;
            }
            for (unsigned i = 0; i != entry.count; ++i) {
              if (entry.tags[i].observation.sequence == at.sequence && entry.tags[i].observation.epoch == at.epoch) {
                entry.tags[i].result_known = true;
                entry.tags[i].successful = successful;
              }
            }
          }
        }
      });
      notify_finish(ui_resources::provider::streamline, at, 0, successful);
    }

    void observe_sl_fg(const stamp &at, unsigned mode, unsigned generated_frames, bool supported, bool successful) noexcept {
      if (sunshine_addon_lifetime::stopping()) {
        return;
      }
      if (!TryAcquireSRWLockExclusive(&lock)) {
        ++dropped;
        return;
      }
      auto *destination = &values.fg.back();
      for (auto &entry : values.fg) {
        if (!entry.present || entry.observation.viewport == at.viewport) {
          destination = &entry;
          break;
        }
      }
      if (destination->present && destination->observation.viewport != at.viewport) {
        values.truncated = true;
      }
      if (!destination->present || destination->observation.epoch != at.epoch || destination->observation.sequence <= at.sequence) {
        *destination = {at, mode, generated_frames, true, supported, successful};
      }
      ReleaseSRWLockExclusive(&lock);
    }

    void observe_ngx(const ngx_evaluation &value) noexcept {
      for (unsigned i = 0; i != std::min(value.parameter_count, maximum_parameters); ++i) {
        const auto &p = value.parameters[i];
        const auto *catalog = ui_resources::find_ngx(p.name);
        if (!catalog) continue;
        resource_observation out;
        out.observation = value.observation;
        out.artifact_id = catalog->artifact_id;
        out.owner = value.owner;
        out.source_id = value.source_id;
        out.feature = value.feature_known ? value.feature : UINT32_MAX;
        out.readable = p.getter_available && p.successful;
        out.descriptor_supported = out.readable && p.type == parameter_type::resource;
        if (out.descriptor_supported) out.native = p.integer;
        // NGX has no declaration of the resource state or this optional
        // resource's active rectangle. Do not borrow the depth/color rectangle.
        notify_resource(out);
      }
      write_frame(value.observation, [&] {
        auto *destination = &values.ngx.back();
        for (auto &entry : values.ngx) {
          if (!entry.observation.session || (entry.owner == value.owner && entry.source_id == value.source_id && entry.handle == value.handle)) {
            destination = &entry;
            break;
          }
        }
        if (destination->observation.session && (destination->owner != value.owner || destination->source_id != value.source_id || destination->handle != value.handle)) {
          values.truncated = true;
        }
        if (!destination->observation.session || destination->observation.sequence <= value.observation.sequence) {
          *destination = value;
        }
      });
    }


    void finish_ngx(const stamp &at, std::uint64_t source_id, bool successful) noexcept {
      write(at, [&] {
        for (auto &entry : values.ngx) {
          if (entry.observation.epoch == at.epoch && entry.observation.sequence == at.sequence && entry.source_id == source_id) {
            entry.result_known = true;
            entry.successful = successful;
          }
        }
      });
      notify_finish(ui_resources::provider::ngx, at, source_id, successful);
    }
  }  // namespace diagnostic

  std::string diagnostic_metadata_json() {
    storage snapshot;
    if (!TryAcquireSRWLockShared(&lock)) {
      return unavailable_snapshot("busy");
    }
    snapshot = values;
    ReleaseSRWLockShared(&lock);
    json result {{"schema", "sunshine-game3d-metadata-1"}, {"status", snapshot.session ? "observed-window" : "not-armed"}, {"diagnostic_session", snapshot.session}, {"armed_tick_ms", snapshot.armed_tick}, {"snapshot_tick_ms", GetTickCount64()}, {"armed", diagnostic_metadata_generation() != 0}, {"dropped_updates", dropped.load()}, {"truncated", snapshot.truncated}, {"pairing", "Independent latest API observations; not necessarily the selected depth or rendered color frame."}, {"success_semantics", "SDK call/getter success only; not capture eligibility, selected source, GPU completion or render readiness."}, {"epoch_semantics", "Adapter observation generation, not an SDK frame number or the consumed frame's provider epoch."}, {"numeric_encoding", "Non-finite floating-point values serialize as null; addresses are identity-only hex strings."}, {"limits", {{"viewports", maximum_viewports}, {"tag_types_per_viewport", maximum_tags}, {"ngx_features", maximum_features}, {"ngx_parameters", maximum_parameters}}}, {"modules", json::array()}, {"streamline", json::array()}, {"frame_generation", json::array()}, {"ngx", json::array()}};
    result["frame_observation_in_window"] = snapshot.frame_observed;
    result["ui_resources"] = ui_inventory(snapshot);
    result["ui_resource_semantics"] = "Fixed verified SDK catalog; unobserved does not mean unsupported. Presence/getter success is not proof of a UI mask, color/depth correspondence or GPU capture. Capture results are supplied separately by the dump owner.";
    for (const auto &m : snapshot.modules) {
      if (m.address) {
        result["modules"].push_back({{"identity", address(m.address)}, {"provider", m.provider}, {"observer_abi", m.abi}, {"path", m.path}, {"path_truncated", m.path_truncated}, {"version_available", m.version_available}, {"file_version", m.version_available ? json(m.file_version) : json(nullptr)}, {"product_version", m.version_available ? json(m.product_version) : json(nullptr)}});
      }
    }
    for (const auto &view : snapshot.sl) {
      if (view.present) {
        json item {{"viewport", view.viewport == UINT32_MAX ? json(nullptr) : json(view.viewport)}, {"truncated", view.truncated}, {"tags", json::array()}, {"constants", view.constants.present ? camera(view.constants) : json(nullptr)}, {"evaluation", nullptr}};
        for (unsigned i = 0; i != view.count; ++i) {
          item["tags"].push_back(tag_json(view.tags[i]));
        }
        if (view.evaluation.present) {
          item["evaluation"] = {{"observation", to_json(view.evaluation.observation)}, {"feature", view.evaluation.feature}, {"successful", view.evaluation.successful}};
        }
        result["streamline"].push_back(std::move(item));
      }
    }
    for (const auto &fg : snapshot.fg) {
      if (fg.present) {
        result["frame_generation"].push_back({{"observation", to_json(fg.observation)}, {"mode", fg.mode}, {"generated_frames", fg.generated_frames}, {"supported", fg.supported}, {"successful", fg.successful}, {"interpretation", "Observed game request, not actual generated presentation count."}});
      }
    }
    for (const auto &value : snapshot.ngx) {
      if (value.observation.session) {
        json item {{"observation", to_json(value.observation)}, {"owner_identity", address(value.owner)}, {"feature_handle_identity", address(value.handle)}, {"source_id", value.source_id}, {"scene_revision", value.feature_known ? json(value.scene_revision) : json(nullptr)}, {"feature_known", value.feature_known}, {"feature", value.feature_known ? json(value.feature) : json(nullptr)}, {"sequence_domain", value.feature_known ? "capture-adapter" : "diagnostic-unknown-feature"}, {"create_width", value.feature_known ? json(value.create_width) : json(nullptr)}, {"create_height", value.feature_known ? json(value.create_height) : json(nullptr)}, {"create_flags", value.feature_known ? json(value.create_flags) : json(nullptr)}, {"truncated", value.truncated}, {"result_known", value.result_known}, {"successful", value.result_known ? json(value.successful) : json(nullptr)}, {"parameters", json::array()}, {"camera_pointer_policy", "Optional pointer presence only; matrix/position memory is not dereferenced or used for rendering."}};
        for (unsigned i = 0; i != std::min(value.parameter_count, maximum_parameters); ++i) {
          const auto &p = value.parameters[i];
          json number = nullptr;
          if (p.getter_available && p.successful) {
            if (p.type == parameter_type::resource || p.type == parameter_type::pointer) {
              number = address(p.integer);
            } else if (p.type == parameter_type::floating) {
              number = p.number;
            } else if (p.type == parameter_type::signed_integer) {
              number = static_cast<std::int64_t>(p.integer);
            } else {
              number = p.integer;
            }
          }
          json row {{"name", p.name}, {"type", unsigned(p.type)}, {"getter_available", p.getter_available}, {"successful", p.getter_available ? json(p.successful) : json(nullptr)}, {"result", p.getter_available ? json(p.result) : json(nullptr)}, {"result_detail", ngx_result(p.result, p.getter_available)}, {"value", number}};
          item["parameters"].push_back(std::move(row));
        }
        result["ngx"].push_back(std::move(item));
      }
    }
    // Reserve space in the outer dump envelope for the actual render/capture
    // frame. Truncation removes whole diagnostic records, never creates invalid
    // JSON or silently cuts a matrix/parameter in half.
    constexpr std::size_t maximum_json_bytes = 192 * 1024;
    result["serialization_omitted"] = {{"sl_tags", 0}, {"ngx_parameters", 0}};
    auto encoded = result.dump(2, ' ', false, json::error_handler_t::replace);
    while (encoded.size() > maximum_json_bytes) {
      bool removed = false;
      for (auto &view : result["streamline"]) {
        if (!view["tags"].empty()) {
          view["tags"].erase(view["tags"].end() - 1);
          view["truncated"] = true;
          result["serialization_omitted"]["sl_tags"] = result["serialization_omitted"]["sl_tags"].get<unsigned>() + 1;
          removed = true;
          break;
        }
      }
      if (!removed) {
        for (auto &feature : result["ngx"]) {
          if (!feature["parameters"].empty()) {
            feature["parameters"].erase(feature["parameters"].end() - 1);
            feature["truncated"] = true;
            result["serialization_omitted"]["ngx_parameters"] = result["serialization_omitted"]["ngx_parameters"].get<unsigned>() + 1;
            removed = true;
            break;
          }
        }
      }
      if (!removed) {
        return unavailable_snapshot("serialization-limit");
      }
      result["truncated"] = true;
      encoded = result.dump(2, ' ', false, json::error_handler_t::replace);
    }
    return encoded;
  }
}  // namespace sunshine_game3d

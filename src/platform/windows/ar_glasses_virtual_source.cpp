/**
 * @file src/platform/windows/ar_glasses_virtual_source.cpp
 * @brief Resolving the owned SudoVDA source and settling its mode, refresh and color.
 */
#include "ar_glasses_virtual_source.h"

#include "ar_glasses.h"
#include "src/logging.h"

#include <thread>
#include <utility>
#include <vector>

namespace ar_glasses {
  using namespace std::literals;
  using namespace internal;

  namespace {
    bool owned_bootstrap_source_identity(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view device_path,
      const active_source_t &source
    ) {
      // A path learned while binding the Add result is stronger than recyclable numeric IDs.
      // Never let a reused LUID/target or an Apollo-like friendly name override that path.
      if (!device_path.empty()) {
        return equal_device_paths(source.device_path, device_path);
      }
      return same_luid(source.target.adapter_id, identity.AdapterLuid) &&
             source.target.target_id == identity.TargetId && is_apollo_ar_virtual_name(source.friendly_name);
    }

    std::optional<std::wstring> refresh_virtual_display_name(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      int expected_width = source_width,
      int expected_height = source_height
    ) {
      const auto resolved = resolve_virtual_display(identity, {}, {}, nullptr, expected_width, expected_height);
      return resolved ? std::optional<std::wstring>(resolved->gdi_name) : std::nullopt;
    }

    bool wait_for_virtual_display_mode(
      std::wstring &display_name,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view physical_device_path,
      bool expected_hdr,
      std::chrono::milliseconds timeout,
      const std::function<bool()> &cancelled
    ) {
      const auto deadline = std::chrono::steady_clock::now() + timeout;
      int stable_observations = 0;
      while (!cancelled() && std::chrono::steady_clock::now() < deadline) {
        bool topology_query_succeeded = false;
        const auto physical_target = find_target(physical_device_path, {}, &topology_query_succeeded);
        if (!topology_query_succeeded) {
          stable_observations = 0;
          std::this_thread::sleep_for(50ms);
          continue;
        }
        if (!physical_target) {
          BOOST_LOG(info) << "AR display disconnected during virtual-display color setup."sv;
          return false;
        }

        const auto refreshed = refresh_virtual_display_name(identity, target_source_width(*physical_target), target_source_height(*physical_target));
        if (!refreshed) {
          stable_observations = 0;
          std::this_thread::sleep_for(50ms);
          continue;
        }
        display_name = *refreshed;

        DEVMODEW mode {};
        const bool geometry_ready = VDISPLAY::getDeviceSettings(display_name.c_str(), mode) &&
                                    mode.dmPelsWidth == target_source_width(*physical_target) &&
                                    mode.dmPelsHeight == target_source_height(*physical_target);
        const auto hdr_state = VDISPLAY::queryDisplayHDR(identity.AdapterLuid, identity.TargetId);
        const bool hdr_matches = hdr_state && *hdr_state == expected_hdr;
        if (geometry_ready && hdr_matches) {
          if (++stable_observations >= 3) {
            return true;
          }
        } else {
          stable_observations = 0;
        }
        for (int sleep_step = 0; sleep_step < 4 && !cancelled(); ++sleep_step) {
          std::this_thread::sleep_for(50ms);
        }
      }
      return false;
    }
  }  // namespace

  namespace internal {
    bool same_virtual_display_identity(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &left,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &right
    ) {
      return same_luid(left.AdapterLuid, right.AdapterLuid) && left.TargetId == right.TargetId;
    }

    virtual_display_probe_t probe_virtual_display(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view device_path,
      std::wstring_view gdi_name,
      int expected_width,
      int expected_height,
      bool require_geometry
    ) {
      const auto snapshot = query_complete_topology_snapshot();
      if (!snapshot) {
        return {};
      }

      auto is_local_virtual_source = [](const active_source_t &source) {
        return is_apollo_ar_virtual_name(source.friendly_name);
      };
      auto select = [&](auto predicate) {
        std::vector<const active_source_t *> matches;
        for (const auto &source : snapshot->sources) {
          if (predicate(source)) {
            matches.push_back(&source);
          }
        }
        return matches;
      };

      auto matches = select([&](const auto &source) {
        return !require_geometry ? owned_bootstrap_source_identity(identity, device_path, source) :
                                   same_luid(source.target.adapter_id, identity.AdapterLuid) &&
                                     source.target.target_id == identity.TargetId &&
                                     is_local_virtual_source(source);
      });
      if (require_geometry && matches.empty() && !device_path.empty()) {
        matches = select([&](const auto &source) {
          // This path was learned from the driver-returned identity while Apollo exclusively
          // owned SudoVDA, so it remains valid when Windows renumbers the target during a mode
          // or Advanced Color transition.
          return equal_device_paths(source.device_path, device_path);
        });
      }
      if (require_geometry && matches.empty() && !gdi_name.empty()) {
        matches = select([&](const auto &source) {
          return equal_device_paths(source.gdi_name, gdi_name) && is_local_virtual_source(source);
        });
      }

      // A complete active snapshot can temporarily renumber every published identifier. Any
      // Apollo AR source proves the driver-owned desktop is still present, but an ambiguous match
      // is deliberately not selected for capture or topology mutation.
      if (require_geometry && matches.empty()) {
        matches = select(is_local_virtual_source);
      }
      if (matches.empty()) {
        return {require_geometry ? virtual_display_presence_e::absent : virtual_display_presence_e::indeterminate, std::nullopt};
      }
      if (matches.size() != 1) {
        return {virtual_display_presence_e::present, std::nullopt};
      }

      const auto &source = *matches.front();
      const bool expected_geometry = source.rect.right - source.rect.left == expected_width &&
                                     source.rect.bottom - source.rect.top == expected_height;
      if (require_geometry && !expected_geometry) {
        // ChangeDisplaySettings can publish an intermediate mode for the retained source. That is
        // present-but-not-ready, never evidence that it is safe to destroy the desktop.
        return {virtual_display_presence_e::present, std::nullopt};
      }

      return {
        virtual_display_presence_e::present,
        resolved_virtual_display_t {
          .identity = {source.target.adapter_id, source.target.target_id},
          .gdi_name = source.gdi_name,
          .device_path = source.device_path,
          .refresh_millihz = source_refresh_millihz(
            snapshot->paths[source.path_index].targetInfo.refreshRate.Numerator,
            snapshot->paths[source.path_index].targetInfo.refreshRate.Denominator
          ),
          .rect = source.rect,
        },
      };
    }

    std::optional<resolved_virtual_display_t> resolve_virtual_display(
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view device_path,
      std::wstring_view gdi_name,
      bool *topology_query_succeeded,
      int expected_width,
      int expected_height,
      bool require_geometry
    ) {
      if (topology_query_succeeded) {
        *topology_query_succeeded = false;
      }
      auto probe = probe_virtual_display(identity, device_path, gdi_name, expected_width, expected_height, require_geometry);
      if (topology_query_succeeded) {
        *topology_query_succeeded = probe.presence != virtual_display_presence_e::indeterminate;
      }
      return std::move(probe.resolved);
    }

    bool configure_virtual_display_hdr(
      std::wstring &display_name,
      const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &identity,
      std::wstring_view physical_device_path,
      bool enable_hdr,
      bool &configured_hdr,
      const std::function<bool()> &cancelled
    ) {
      configured_hdr = false;
      // Match the remote-stream workaround: wait for the new mode, force Advanced Color off,
      // then enable it after the first transition has settled. Windows often ignores a direct
      // enable immediately after IddCx output creation or a resolution/topology change.
      if (!wait_for_virtual_display_mode(display_name, identity, physical_device_path, false, 5s, cancelled)) {
        if (cancelled()) {
          return false;
        }
        BOOST_LOG(warning) << "Local AR virtual display did not settle before HDR configuration."sv;
      }
      if (!VDISPLAY::setDisplayHDR(identity.AdapterLuid, identity.TargetId, false)) {
        const auto current_hdr = VDISPLAY::queryDisplayHDR(identity.AdapterLuid, identity.TargetId);
        if (!current_hdr || *current_hdr) {
          BOOST_LOG(error) << "Could not authoritatively reset the local AR virtual display to SDR before HDR setup."sv;
          return false;
        }
      }
      if (!wait_for_virtual_display_mode(display_name, identity, physical_device_path, false, 5s, cancelled)) {
        if (cancelled()) {
          return false;
        }
        BOOST_LOG(error) << "Local AR virtual display did not reach a stable SDR state."sv;
        return false;
      }
      if (!enable_hdr) {
        return true;
      }

      if (!VDISPLAY::setDisplayHDR(identity.AdapterLuid, identity.TargetId, true)) {
        BOOST_LOG(warning) << "Windows rejected HDR for the local AR virtual display; using color-managed SDR presentation."sv;
        return true;
      }
      if (!wait_for_virtual_display_mode(display_name, identity, physical_device_path, true, 15s, cancelled)) {
        if (cancelled()) {
          return false;
        }
        BOOST_LOG(warning) << "Local AR virtual display did not reach a stable HDR state; using color-managed SDR presentation."sv;
        if (!VDISPLAY::setDisplayHDR(identity.AdapterLuid, identity.TargetId, false) || !wait_for_virtual_display_mode(display_name, identity, physical_device_path, false, 5s, cancelled)) {
          BOOST_LOG(error) << "Local AR virtual display could not recover to SDR after HDR setup failed."sv;
          return false;
        }
        return true;
      }
      configured_hdr = true;
      return true;
    }
  }  // namespace internal

#ifdef SUNSHINE_TESTS
  std::optional<int> detail::configure_local_source_refresh_for_test(
    int requested_millihz,
    const std::function<std::optional<int>()> &query,
    const std::function<void()> &apply,
    const std::function<void(std::chrono::milliseconds)> &wait,
    const std::function<bool()> &cancelled
  ) {
    return configure_source_refresh_with_io(
      requested_millihz,
      [&]() -> std::optional<resolved_virtual_display_t> {
        const auto measured = query();
        if (!measured) {
          return std::nullopt;
        }
        return resolved_virtual_display_t {.refresh_millihz = *measured};
      },
      [&](const resolved_virtual_display_t &) {
        apply();
      },
      wait,
      cancelled
    );
  }

  bool detail::establish_local_source_mode_for_test(
    int width,
    int height,
    const std::function<std::optional<local_source_mode_t>()> &query,
    const std::function<void()> &apply,
    const std::function<void(std::chrono::milliseconds)> &wait,
    const std::function<std::chrono::steady_clock::time_point()> &now,
    const std::function<bool()> &cancelled
  ) {
    return establish_source_mode_with_io(
      width,
      height,
      [&]() -> std::optional<resolved_virtual_display_t> {
        const auto mode = query();
        if (!mode) {
          return std::nullopt;
        }
        return resolved_virtual_display_t {
          .refresh_millihz = mode->refresh_millihz,
          .rect = {0, 0, mode->width, mode->height},
        };
      },
      [&](const resolved_virtual_display_t &) {
        apply();
      },
      wait,
      now,
      cancelled
    );
  }

  bool detail::settle_local_source_color_for_test(
    int width,
    int height,
    int refresh_millihz,
    bool expected_hdr,
    std::chrono::milliseconds timeout,
    const std::function<std::optional<local_source_color_t>()> &query,
    const std::function<void()> &set_mode,
    const std::function<void(bool)> &set_color,
    const std::function<void(std::chrono::milliseconds)> &wait,
    const std::function<std::chrono::steady_clock::time_point()> &now,
    const std::function<bool()> &cancelled
  ) {
    return settle_source_color_with_io(
      width,
      height,
      refresh_millihz,
      expected_hdr,
      timeout,
      [&]() -> std::optional<source_color_state_t> {
        const auto source = query();
        if (!source) {
          return std::nullopt;
        }
        return source_color_state_t {
          resolved_virtual_display_t {
            .refresh_millihz = source->mode.refresh_millihz,
            .rect = {0, 0, source->mode.width, source->mode.height},
          },
          source->hdr,
        };
      },
      [&](const resolved_virtual_display_t &) {
        set_mode();
      },
      [&](const resolved_virtual_display_t &, bool hdr) {
        set_color(hdr);
      },
      wait,
      now,
      cancelled
    );
  }

  bool detail::bootstrap_source_identity_matches_for_test(
    const virtual_display_identity_contract_t &owned,
    const virtual_display_identity_contract_t &observed
  ) {
    active_source_t source;
    source.target = {observed.adapter_id, observed.target_id};
    source.device_path = observed.device_path;
    source.gdi_name = observed.gdi_name;
    source.friendly_name = observed.friendly_name;
    return owned_bootstrap_source_identity(
      {owned.adapter_id, owned.target_id},
      owned.device_path,
      source
    );
  }
#endif
}  // namespace ar_glasses

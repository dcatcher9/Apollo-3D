/**
 * @file src/platform/windows/ar_glasses_virtual_source.cpp
 * @brief Resolving the owned SudoVDA source and settling its mode, refresh and color.
 */
#include "ar_glasses_virtual_source.h"

#include "ar_glasses.h"
#include "src/logging.h"

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

    void log_color_settle(std::string_view display, const color_settle_request_t &request, const color_settle_result_t &result) {
      const auto color = request.hdr ? "HDR"sv : "SDR"sv;
      switch (result.outcome) {
        case color_settle_e::settled:
          BOOST_LOG(info) << display << " color settled: "sv << color << " after "sv << result.elapsed.count()
                          << " ms, "sv << result.color_requests << " request(s), "sv << result.mode_repairs << " mode repair(s)."sv;
          return;
        case color_settle_e::rejected:
          BOOST_LOG(warning) << display << " color rejected: Windows refused all "sv << result.refused_requests << ' '
                             << color << " requests."sv;
          return;
        case color_settle_e::timed_out:
          BOOST_LOG(warning) << display << " color did not settle: "sv << color << " was not reported within "sv
                             << request.budget.count() << " ms, after "sv << result.color_requests << " request(s) ("sv
                             << result.refused_requests << " refused) and "sv << result.mode_repairs << " mode repair(s)."sv;
          return;
        case color_settle_e::disconnected:
        case color_settle_e::cancelled:
          return;
      }
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

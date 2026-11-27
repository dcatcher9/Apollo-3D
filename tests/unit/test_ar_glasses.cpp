
#include "src/platform/windows/ar_glasses.h"
#include "src/platform/windows/rayneo_wear_monitor.h"
#include "src/platform/windows/virtual_display.h"

#include <array>
#include <chrono>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <nlohmann/json.hpp>
#include <pbt.h>
#include <thread>

using namespace std::chrono_literals;

TEST(LocalMonitorDesktopGuard, LockedSessionOnDefaultDesktopRequiresUnlockingWindows) {
  const auto reason = ar_glasses::detail::monitor_desktop_block_reason_for_test(true, L"Default");
  EXPECT_NE(reason.find("Unlock Windows"), std::string::npos);
}

TEST(LocalMonitorDesktopGuard, UnlockedDefaultDesktopMayStartOrContinuePresentation) {
  EXPECT_TRUE(ar_glasses::detail::monitor_desktop_block_reason_for_test(false, L"Default").empty());
  EXPECT_TRUE(ar_glasses::detail::monitor_desktop_block_reason_for_test(false, L"default", L"DEFAULT").empty());
}

TEST(LocalMonitorDesktopGuard, VisibleLockOverlayBlocksAnApparentlyUnlockedDefaultDesktop) {
  const auto reason = ar_glasses::detail::monitor_desktop_block_reason_for_test(false, L"Default", L"Default", true);
  EXPECT_NE(reason.find("Dismiss the Windows lock or sign-in screen"), std::string::npos);
}

TEST(LocalMonitorDesktopGuard, UnknownSessionStateCannotCertifyAnInteractiveDesktop) {
  const auto reason = ar_glasses::detail::monitor_desktop_block_reason_for_test(std::nullopt, L"Default");
  EXPECT_NE(reason.find("could not be verified"), std::string::npos);
}

TEST(LocalMonitorDesktopGuard, SecureOrUnavailableDesktopsCannotStartOrContinuePresentation) {
  EXPECT_FALSE(ar_glasses::detail::monitor_desktop_block_reason_for_test(false, L"Winlogon").empty());
  EXPECT_FALSE(ar_glasses::detail::monitor_desktop_block_reason_for_test(false, L"Default", L"Winlogon").empty());
  EXPECT_FALSE(ar_glasses::detail::monitor_desktop_block_reason_for_test(false, L"Winlogon", L"Winlogon").empty());
  EXPECT_FALSE(ar_glasses::detail::monitor_desktop_block_reason_for_test(false, L"").empty());
}

TEST(LocalMonitor, NativePhysicalModesAreIndependentOfArPacking) {
  using enum ar_glasses::presentation_mode_e;
  EXPECT_EQ(ar_glasses::classify_monitor_mode(5120, 2160, 165000, false), monitor_host);
  EXPECT_EQ(ar_glasses::classify_monitor_mode(2560, 1080, 330000, true), monitor_game);
  EXPECT_EQ(ar_glasses::classify_mode(5120, 2160), unsupported);
  EXPECT_EQ(ar_glasses::classify_mode(3840, 1080), sbs_ai);
  EXPECT_EQ(ar_glasses::classify_monitor_mode(8193, 2160, 165000, false), unsupported);
  EXPECT_EQ(ar_glasses::classify_monitor_mode(1920, 1080, 0, false), unsupported);
  EXPECT_EQ(ar_glasses::classify_monitor_mode(0, 1080, 120000, false), unsupported);
}

TEST(LocalMonitor, NativeResolutionChangeRequiresNewSourceButRefreshCanReconfigure) {
  ar_glasses::detail::local_session_contract_t before;
  before.device_path = L"monitor-instance";
  before.rect = {0, 0, 2560, 1080};
  before.mode = ar_glasses::presentation_mode_e::monitor_host;
  auto after = before;
  after.refresh_millihz = 165000;
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after, true));
  after.rect = {0, 0, 5120, 2160};
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after, true));
  after = before;
  after.mode = ar_glasses::presentation_mode_e::sbs_ai;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after, true));
}

namespace {
  using local_session_contract_t = ar_glasses::detail::local_session_contract_t;

  local_session_contract_t monitor_source_contract() {
    local_session_contract_t target;
    target.device_path = LR"(\\?\DISPLAY#TCL03D4#monitor)";
    target.adapter_id = {42, 7};
    target.mode = ar_glasses::presentation_mode_e::monitor_host;
    target.hdr_known = true;
    target.hdr_supported = true;
    target.hdr_active = true;
    target.rect = {1920, -1080, 3840, 0};
    target.refresh_millihz = 120000;
    target.source_adapter_id = {42, 7};
    target.source_id = 1;
    target.target_id = 2;
    target.gdi_name = R"(\\.\DISPLAY2)";
    target.desktop_topology = "primary=owned-source;neighbor@3840,0:1920x1080@60000";
    target.hdr_user_enabled = true;
    target.hdr_bits_per_color = 10;
    return target;
  }
}  // namespace

TEST(LocalMonitorSourceSwitch, ExactMonitorContractAllowsEitherProviderDirection) {
  using ar_glasses::detail::local_monitor_source_only_change_for_test;
  for (const bool hdr : {false, true}) {
    SCOPED_TRACE(hdr);
    auto host = monitor_source_contract();
    host.hdr_active = hdr;
    host.hdr_user_enabled = hdr;
    host.hdr_bits_per_color = hdr ? 10u : 8u;
    auto game = host;
    game.mode = ar_glasses::presentation_mode_e::monitor_game;

    EXPECT_TRUE(local_monitor_source_only_change_for_test(host, game));
    EXPECT_TRUE(local_monitor_source_only_change_for_test(game, host));
  }
}

TEST(LocalMonitorSourceSwitch, UnchangedProviderAndOtherSurfaceModesDoNotQualify) {
  using enum ar_glasses::presentation_mode_e;
  using ar_glasses::detail::local_monitor_source_only_change_for_test;
  const auto host = monitor_source_contract();
  auto other = host;
  for (auto mode : {monitor_host, normal, sbs_ai, unsupported}) {
    SCOPED_TRACE(static_cast<int>(mode));
    other.mode = mode;
    EXPECT_FALSE(local_monitor_source_only_change_for_test(host, other));
    EXPECT_FALSE(local_monitor_source_only_change_for_test(other, host));
  }
}

TEST(LocalMonitorSourceSwitch, GeometryIdentityOrColorRevisionsRequireDisplayRepair) {
  using ar_glasses::detail::local_monitor_source_only_change_for_test;
  const auto host = monitor_source_contract();
  auto game = host;
  game.mode = ar_glasses::presentation_mode_e::monitor_game;

  const auto expect_rejected_change = [&](const char *name, auto member, const auto &value) {
    SCOPED_TRACE(name);
    auto changed = game;
    changed.*member = value;
    EXPECT_FALSE(local_monitor_source_only_change_for_test(host, changed));
  };
  expect_rejected_change("physical device", &local_session_contract_t::device_path, LR"(\\?\DISPLAY#other)");
  expect_rejected_change("target adapter", &local_session_contract_t::adapter_id, LUID {43, 7});
  expect_rejected_change("target identity", &local_session_contract_t::target_id, 3u);
  expect_rejected_change("source adapter", &local_session_contract_t::source_adapter_id, LUID {42, 8});
  expect_rejected_change("source identity", &local_session_contract_t::source_id, 2u);
  expect_rejected_change("GDI name", &local_session_contract_t::gdi_name, R"(\\.\DISPLAY3)");
  expect_rejected_change("moved sink", &local_session_contract_t::rect, RECT {1940, -1110, 3860, -30});
  expect_rejected_change("raster", &local_session_contract_t::rect, RECT {1920, -1080, 4480, 0});
  expect_rejected_change("refresh", &local_session_contract_t::refresh_millihz, 119880);
  expect_rejected_change("other display moved", &local_session_contract_t::desktop_topology, "primary=owned-source;neighbor@3840,100:1920x1080@60000");
  expect_rejected_change("primary sink", &local_session_contract_t::is_primary, true);
  expect_rejected_change("cloned sink", &local_session_contract_t::is_cloned, true);
  expect_rejected_change("unknown HDR", &local_session_contract_t::hdr_known, false);
  expect_rejected_change("HDR support", &local_session_contract_t::hdr_supported, false);
  expect_rejected_change("HDR setting", &local_session_contract_t::hdr_user_enabled, false);
  expect_rejected_change("active HDR", &local_session_contract_t::hdr_active, false);
  expect_rejected_change("HDR policy", &local_session_contract_t::hdr_limited_by_policy, true);
  expect_rejected_change("bit depth", &local_session_contract_t::hdr_bits_per_color, 8u);

  auto unverified_host = host;
  unverified_host.hdr_known = false;
  game.hdr_known = false;
  EXPECT_FALSE(local_monitor_source_only_change_for_test(unverified_host, game));
}

TEST(LocalMonitorSourceSwitch, RechecksAuthorityAroundVerificationWithoutStoppingScanout) {
  using ar_glasses::detail::verify_monitor_source_request_for_test;
  enum class event_e {
    none,
    remote_busy,
    request_superseded,
    desktop_locked,
    topology_changed,
  };

  struct scenario_t {
    event_e event;
    const char *trace;
    bool verified;
  };

  // R checks the live request and V verifies the existing desktop. Either side can revoke it.
  const scenario_t scenarios[] {
    {event_e::none, "RVR", true},
    {event_e::remote_busy, "R", false},
    {event_e::request_superseded, "RVR", false},
    {event_e::desktop_locked, "RV", false},
    {event_e::topology_changed, "RVR", false},
  };
  for (const auto &scenario : scenarios) {
    SCOPED_TRACE(static_cast<int>(scenario.event));
    bool current = scenario.event != event_e::remote_busy;
    std::string trace;
    const bool verified = verify_monitor_source_request_for_test(
      [&] {
        trace += 'R';
        return current;
      },
      [&] {
        trace += 'V';
        current = scenario.event != event_e::topology_changed &&
                  scenario.event != event_e::request_superseded;
        return scenario.event != event_e::desktop_locked;
      }
    );
    EXPECT_EQ(verified, scenario.verified);
    EXPECT_EQ(trace, scenario.trace);
  }
}

namespace {
  std::array<std::uint8_t, 65> rayneo_sensor_report(
    float proximity,
    std::uint32_t tick = 1,
    std::uint8_t fill = 0
  ) {
    std::array<std::uint8_t, 65> report {};
    report.fill(fill);
    report[0] = 0x00;
    report[1] = 0x99;
    report[2] = 0x65;
    std::memcpy(report.data() + 1 + 40, &tick, sizeof(tick));
    std::memcpy(report.data() + 1 + 44, &proximity, sizeof(proximity));
    return report;
  }

  std::array<std::uint8_t, 65> rayneo_board_info_ack(std::uint8_t board_id) {
    std::array<std::uint8_t, 65> report {};
    report[0] = 0x00;
    report[1] = 0x99;
    report[2] = 0xC8;
    report[1 + 8] = 0x00;
    report[1 + 21] = board_id;
    return report;
  }

  nlohmann::json legacy_recovery_record(
    std::string device_path = R"(\\?\DISPLAY#TCL03D4#test)"
  ) {
    return {
      {"device_path", std::move(device_path)},
      {"original_left", 5120},
      {"original_top", 0},
      {"original_right", 7040},
      {"original_bottom", 1080},
      {"applied_left", 5119},
      {"applied_top", 2160},
      {"applied_right", 7039},
      {"applied_bottom", 3240},
    };
  }

  nlohmann::json rect_json(LONG left, LONG top, LONG right, LONG bottom) {
    return {
      {"left", left},
      {"top", top},
      {"right", right},
      {"bottom", bottom},
    };
  }

  nlohmann::json transactional_recovery_record(
    nlohmann::json owned_rects,
    nlohmann::json pending_rect = nullptr
  ) {
    return {
      {"device_path", R"(\\?\DISPLAY#TCL03D4#test)"},
      {"original_rect", rect_json(5120, 0, 7040, 1080)},
      {"owned_rects", std::move(owned_rects)},
      {"pending_rect", std::move(pending_rect)},
    };
  }

  nlohmann::json current_recovery_record() {
    auto owned = nlohmann::json::array();
    owned.emplace_back(rect_json(7040, 0, 8960, 1080));
    return transactional_recovery_record(std::move(owned));
  }

  nlohmann::json recovery_document(nlohmann::json recoveries, int version = 4) {
    return {
      {"version", version},
      {"recoveries", std::move(recoveries)},
    };
  }

  struct source_refresh_probe_t {
    std::optional<int> measured = 120000;
    std::optional<int> after_set;
    unsigned queries = 0;
    unsigned setters = 0;
    unsigned waits = 0;
    std::chrono::milliseconds elapsed {};
    bool cancelled = false;
    bool cancel_on_wait = false;

    std::optional<int> configure(int requested) {
      return ar_glasses::detail::configure_local_source_refresh_for_test(
        requested,
        [&]() {
          ++queries;
          return measured;
        },
        [&]() {
          ++setters;
          if (after_set) {
            measured = after_set;
          }
        },
        [&](std::chrono::milliseconds delay) {
          ++waits;
          elapsed += delay;
          cancelled = cancelled || cancel_on_wait;
        },
        [&]() {
          return cancelled;
        }
      );
    }
  };

  struct source_mode_probe_t {
    using mode_t = ar_glasses::detail::local_source_mode_t;
    std::optional<mode_t> measured = mode_t {1920, 1080, 60000};
    std::optional<mode_t> after_set;
    unsigned visible_after_waits = 0;
    unsigned queries = 0;
    unsigned setters = 0;
    unsigned waits = 0;
    std::chrono::milliseconds elapsed {};
    std::chrono::milliseconds setter_duration {};
    bool cancelled = false;
    bool cancel_on_query = false;
    bool cancel_on_set = false;
    bool cancel_on_wait = false;

    bool establish() {
      return ar_glasses::detail::establish_local_source_mode_for_test(
        3840,
        2160,
        [&]() -> std::optional<mode_t> {
          ++queries;
          cancelled = cancelled || cancel_on_query;
          return waits >= visible_after_waits ? measured : std::nullopt;
        },
        [&]() {
          ++setters;
          elapsed += setter_duration;
          if (after_set) {
            measured = after_set;
          }
          cancelled = cancelled || cancel_on_set;
        },
        [&](std::chrono::milliseconds delay) {
          ++waits;
          elapsed += delay;
          cancelled = cancelled || cancel_on_wait;
        },
        [&]() {
          return std::chrono::steady_clock::time_point {} + elapsed;
        },
        [&]() {
          return cancelled;
        }
      );
    }
  };

  struct source_color_probe_t {
    using color_t = ar_glasses::detail::local_source_color_t;
    std::optional<color_t> measured = color_t {{3840, 2160, 120000}, false};
    std::chrono::milliseconds elapsed {};
    std::chrono::milliseconds visible_after {};
    unsigned queries = 0;
    unsigned mode_sets = 0;
    unsigned color_sets = 0;
    bool accept_mode = true;
    bool accept_color = true;
    bool enable_hdr_resets_mode_once = false;
    bool enable_hdr_always_resets_mode = false;
    bool mode_resets_hdr = false;
    bool physical_sink_active = true;
    bool cancelled = false;
    bool cancel_on_query = false;
    bool cancel_on_mode = false;
    bool cancel_on_color = false;
    bool cancel_on_wait = false;

    bool settle(bool hdr, std::chrono::milliseconds timeout = 5s) {
      return ar_glasses::detail::settle_local_source_color_for_test(
        3840,
        2160,
        120000,
        hdr,
        timeout,
        [&]() -> std::optional<color_t> {
          ++queries;
          cancelled = cancelled || cancel_on_query;
          return elapsed >= visible_after ? measured : std::nullopt;
        },
        [&]() {
          ++mode_sets;
          if (accept_mode && measured) {
            measured->mode = {3840, 2160, 120000};
            if (mode_resets_hdr) {
              measured->hdr = false;
            }
          }
          cancelled = cancelled || cancel_on_mode;
        },
        [&](bool enabled) {
          ++color_sets;
          if (accept_color && measured) {
            measured->hdr = enabled;
            if (enabled && (enable_hdr_resets_mode_once || enable_hdr_always_resets_mode)) {
              measured->mode = {1920, 1080, 60000};
              physical_sink_active = false;
              enable_hdr_resets_mode_once = false;
            }
          }
          cancelled = cancelled || cancel_on_color;
        },
        [&](std::chrono::milliseconds delay) {
          elapsed += delay;
          cancelled = cancelled || cancel_on_wait;
        },
        [&]() {
          return std::chrono::steady_clock::time_point {} + elapsed;
        },
        [&]() {
          return cancelled;
        }
      );
    }
  };
}  // namespace

TEST(LocalMonitorSourceColor, RepairsTheObservedHdrModeResetBeforeOwnerRestoresInactiveTv) {
  source_color_probe_t probe;
  probe.enable_hdr_resets_mode_once = true;
  EXPECT_TRUE(probe.settle(true, 15s));
  ASSERT_TRUE(probe.measured);
  EXPECT_EQ(probe.measured->mode.width, 3840);
  EXPECT_EQ(probe.measured->mode.height, 2160);
  EXPECT_EQ(probe.measured->mode.refresh_millihz, 120000);
  EXPECT_EQ(probe.measured->hdr, true);
  EXPECT_EQ(probe.mode_sets, 1u);
  EXPECT_EQ(probe.color_sets, 1u);
  EXPECT_EQ(probe.elapsed, 500ms);
  // This source-only phase deliberately leaves scanout restoration to the outer owner
  // transaction, whose result must pass the final physical contract before desktop handoff.
  EXPECT_FALSE(probe.physical_sink_active);
}

TEST(LocalMonitorSourceColor, RequiresRequestedCadenceEvenWhenRasterAndHdrAlreadyMatch) {
  source_color_probe_t probe;
  probe.measured = source_color_probe_t::color_t {{3840, 2160, 60000}, true};
  EXPECT_TRUE(probe.settle(true));
  EXPECT_EQ(probe.mode_sets, 1u);
  EXPECT_EQ(probe.color_sets, 0u);
}

TEST(LocalMonitorSourceColor, RepairsColorAgainIfTheNativeModeSetterResetsHdr) {
  source_color_probe_t probe;
  probe.measured = source_color_probe_t::color_t {{1920, 1080, 60000}, true};
  probe.mode_resets_hdr = true;
  EXPECT_TRUE(probe.settle(true));
  EXPECT_EQ(probe.mode_sets, 1u);
  EXPECT_EQ(probe.color_sets, 1u);
  EXPECT_EQ(probe.measured->hdr, true);
}

TEST(LocalMonitorSourceColor, WaitsForAuthoritativeSourceAndStableColorObservations) {
  source_color_probe_t probe;
  probe.visible_after = 2s;
  probe.measured->hdr = true;
  EXPECT_TRUE(probe.settle(true));
  EXPECT_EQ(probe.elapsed, 2400ms);
  EXPECT_EQ(probe.mode_sets, 0u);
  EXPECT_EQ(probe.color_sets, 0u);
}

TEST(LocalMonitorSourceColor, UnknownIdentityOrColorCannotCertifyOrReceiveColorSetters) {
  source_color_probe_t missing;
  missing.measured.reset();
  EXPECT_FALSE(missing.settle(true));
  EXPECT_EQ(missing.mode_sets, 0u);
  EXPECT_EQ(missing.color_sets, 0u);
  EXPECT_EQ(missing.elapsed, 5s);

  source_color_probe_t unknown_hdr;
  unknown_hdr.measured->hdr.reset();
  EXPECT_FALSE(unknown_hdr.settle(true));
  EXPECT_EQ(unknown_hdr.mode_sets, 0u);
  EXPECT_EQ(unknown_hdr.color_sets, 0u);
}

TEST(LocalMonitorSourceColor, RejectedHdrFailsWithinTheDeadlineWithoutAnSdrFallback) {
  source_color_probe_t probe;
  probe.accept_color = false;
  EXPECT_FALSE(probe.settle(true));
  EXPECT_EQ(probe.mode_sets, 0u);
  EXPECT_EQ(probe.color_sets, 3u);
  EXPECT_EQ(probe.measured->hdr, false);
  EXPECT_EQ(probe.elapsed, 5s);
}

TEST(LocalMonitorSourceColor, RepeatedModeAndColorResetsCannotRetryIndefinitely) {
  source_color_probe_t probe;
  probe.enable_hdr_always_resets_mode = true;
  probe.mode_resets_hdr = true;
  EXPECT_FALSE(probe.settle(true));
  EXPECT_EQ(probe.mode_sets, 3u);
  EXPECT_EQ(probe.color_sets, 3u);
  EXPECT_EQ(probe.elapsed, 5s);
}

TEST(LocalMonitorSourceColor, CancellationStopsQueriesSettersAndThePublicationWait) {
  source_color_probe_t before_query;
  before_query.cancelled = true;
  EXPECT_FALSE(before_query.settle(true));
  EXPECT_EQ(before_query.queries, 0u);

  source_color_probe_t during_query;
  during_query.cancel_on_query = true;
  EXPECT_FALSE(during_query.settle(true));
  EXPECT_EQ(during_query.color_sets, 0u);

  source_color_probe_t during_mode;
  during_mode.measured->mode.refresh_millihz = 60000;
  during_mode.cancel_on_mode = true;
  EXPECT_FALSE(during_mode.settle(true));
  EXPECT_EQ(during_mode.mode_sets, 1u);
  EXPECT_EQ(during_mode.color_sets, 0u);
  EXPECT_EQ(during_mode.elapsed, 0ms);

  source_color_probe_t during_color;
  during_color.cancel_on_color = true;
  EXPECT_FALSE(during_color.settle(true));
  EXPECT_EQ(during_color.color_sets, 1u);
  EXPECT_EQ(during_color.elapsed, 0ms);

  source_color_probe_t during_wait;
  during_wait.cancel_on_wait = true;
  EXPECT_FALSE(during_wait.settle(true));
  EXPECT_EQ(during_wait.elapsed, 50ms);
}

TEST(LocalMonitorBootstrap, RepairsRememberedArGeometryBeforeRequiringNativeReadback) {
  source_mode_probe_t probe;
  probe.after_set = source_mode_probe_t::mode_t {3840, 2160, 120000};
  EXPECT_TRUE(probe.establish());
  EXPECT_EQ(probe.queries, 2u);
  EXPECT_EQ(probe.setters, 1u);
  EXPECT_EQ(probe.elapsed, 50ms);
}

TEST(LocalMonitorBootstrap, WaitsForOwnedIdentityBeforeTheFirstModeSetter) {
  source_mode_probe_t probe;
  probe.visible_after_waits = 40;
  probe.after_set = source_mode_probe_t::mode_t {3840, 2160, 120000};
  EXPECT_TRUE(probe.establish());
  EXPECT_EQ(probe.setters, 1u);
  EXPECT_EQ(probe.elapsed, 2050ms);
}

TEST(LocalMonitorBootstrap, NativeGeometryDoesNotReplaceTheLaterCadenceRepair) {
  source_mode_probe_t probe;
  probe.measured = source_mode_probe_t::mode_t {3840, 2160, 60000};
  EXPECT_TRUE(probe.establish());
  EXPECT_EQ(probe.setters, 0u);
  EXPECT_EQ(probe.waits, 0u);
}

TEST(LocalMonitorBootstrap, MatchingRefreshCannotCertifyTheWrongRaster) {
  source_mode_probe_t probe;
  probe.measured = source_mode_probe_t::mode_t {1920, 1080, 120000};
  EXPECT_FALSE(probe.establish());
  EXPECT_EQ(probe.setters, 5u);
  EXPECT_EQ(probe.elapsed, 5s);
}

TEST(LocalMonitorBootstrap, MissingIdentityAndUnknownCadenceRemainBoundedAndUnusable) {
  source_mode_probe_t missing;
  missing.measured.reset();
  EXPECT_FALSE(missing.establish());
  EXPECT_EQ(missing.setters, 0u);
  EXPECT_EQ(missing.elapsed, 5s);

  source_mode_probe_t unknown_cadence;
  unknown_cadence.measured = source_mode_probe_t::mode_t {3840, 2160, 0};
  EXPECT_FALSE(unknown_cadence.establish());
  EXPECT_EQ(unknown_cadence.setters, 5u);
  EXPECT_EQ(unknown_cadence.elapsed, 5s);
}

TEST(LocalMonitorBootstrap, CancellationCannotMutateOrCertifyAfterAnObservation) {
  source_mode_probe_t before_query;
  before_query.cancelled = true;
  EXPECT_FALSE(before_query.establish());
  EXPECT_EQ(before_query.queries, 0u);

  source_mode_probe_t during_query;
  during_query.cancel_on_query = true;
  EXPECT_FALSE(during_query.establish());
  EXPECT_EQ(during_query.setters, 0u);
  EXPECT_EQ(during_query.waits, 0u);

  source_mode_probe_t during_set;
  during_set.cancel_on_set = true;
  during_set.after_set = source_mode_probe_t::mode_t {3840, 2160, 120000};
  EXPECT_FALSE(during_set.establish());
  EXPECT_EQ(during_set.setters, 1u);
  EXPECT_EQ(during_set.waits, 0u);

  source_mode_probe_t during_wait;
  during_wait.cancel_on_wait = true;
  EXPECT_FALSE(during_wait.establish());
  EXPECT_EQ(during_wait.elapsed, 50ms);
}

TEST(LocalMonitorBootstrap, SlowModeSettersConsumeThePublicationDeadline) {
  source_mode_probe_t probe;
  probe.setter_duration = 5s;
  probe.after_set = source_mode_probe_t::mode_t {3840, 2160, 120000};
  EXPECT_FALSE(probe.establish());
  EXPECT_EQ(probe.setters, 1u);
  EXPECT_EQ(probe.waits, 0u);
}

TEST(LocalMonitorBootstrapIdentity, RequiresDriverIdentityBeforeLearningADevicePath) {
  ar_glasses::detail::virtual_display_identity_contract_t owned;
  owned.adapter_id = {42, 0};
  owned.target_id = 7;
  auto observed = owned;
  observed.device_path = L"owned-device-path";
  observed.gdi_name = L"\\\\.\\DISPLAY5";
  observed.friendly_name = L"Apollo AR Desktop";
  EXPECT_TRUE(ar_glasses::detail::bootstrap_source_identity_matches_for_test(owned, observed));
  observed.target_id = 8;
  EXPECT_FALSE(ar_glasses::detail::bootstrap_source_identity_matches_for_test(owned, observed));
  observed.target_id = 7;
  observed.friendly_name = L"Ordinary monitor";
  EXPECT_FALSE(ar_glasses::detail::bootstrap_source_identity_matches_for_test(owned, observed));
}

TEST(LocalMonitorBootstrapIdentity, LearnedPathOverridesRecycledNumericIdentityAndFriendlyNames) {
  ar_glasses::detail::virtual_display_identity_contract_t owned;
  owned.adapter_id = {42, 0};
  owned.target_id = 7;
  owned.device_path = L"owned-device-path";
  auto observed = owned;
  observed.friendly_name = L"Apollo AR Desktop";
  observed.device_path = L"different-device-path";
  EXPECT_FALSE(ar_glasses::detail::bootstrap_source_identity_matches_for_test(owned, observed));
  observed = owned;
  observed.adapter_id = {123, 0};
  observed.target_id = 99;
  EXPECT_TRUE(ar_glasses::detail::bootstrap_source_identity_matches_for_test(owned, observed));
}

TEST(ArGlassesMode, SelectsNormalForNativeTwoDimensionalMode) {
  EXPECT_EQ(
    ar_glasses::classify_mode(1920, 1080),
    ar_glasses::presentation_mode_e::normal
  );
}

TEST(ArGlassesMode, SelectsFullSbsForDoubleWidthMode) {
  EXPECT_EQ(
    ar_glasses::classify_mode(3840, 1080),
    ar_glasses::presentation_mode_e::sbs_ai
  );
}

TEST(ArGlassesMode, RejectsUnrecognizedModes) {
  EXPECT_EQ(
    ar_glasses::classify_mode(2560, 1080),
    ar_glasses::presentation_mode_e::unsupported
  );
  EXPECT_EQ(
    ar_glasses::classify_mode(3840, 2160),
    ar_glasses::presentation_mode_e::unsupported
  );
}

TEST(ArGlassesSourceRefresh, PreservesFractionalCadenceAndDistinguishesAHalfRateSource) {
  EXPECT_EQ(ar_glasses::detail::local_source_refresh_millihz_for_test(60000, 1001), 59940);
  EXPECT_EQ(ar_glasses::detail::local_source_refresh_millihz_for_test(120000, 1001), 119880);
  EXPECT_EQ(ar_glasses::detail::local_source_refresh_millihz_for_test(60, 1), 60000);
  EXPECT_EQ(ar_glasses::detail::local_source_refresh_millihz_for_test(120, 1), 120000);
}

TEST(ArGlassesSourceRefresh, RejectsUnknownAndOverflowingRefreshEvidence) {
  EXPECT_EQ(ar_glasses::detail::local_source_refresh_millihz_for_test(0, 1), 0);
  EXPECT_EQ(ar_glasses::detail::local_source_refresh_millihz_for_test(60, 0), 0);
  EXPECT_EQ(ar_glasses::detail::local_source_refresh_millihz_for_test(std::numeric_limits<std::uint32_t>::max(), 1), 0);
  EXPECT_EQ(ar_glasses::detail::local_source_refresh_millihz_for_test(120000000, 1000000), 120000);
}

TEST(ArGlassesSourceRefresh, TinyClockDifferenceBypassesModeSettersAndPollingWithoutRounding) {
  for (const auto rates : std::array {
         std::array {120000, 120013},
         std::array {120013, 120000},
         std::array {120000, 120024},
         std::array {std::numeric_limits<int>::max(), std::numeric_limits<int>::max() - 1},
       }) {
    source_refresh_probe_t probe;
    probe.measured = rates[0];
    EXPECT_EQ(probe.configure(rates[1]), rates[0]);
    EXPECT_EQ(probe.measured, rates[0]);
    EXPECT_EQ(probe.queries, 1u);
    EXPECT_EQ(probe.setters, 0u);
    EXPECT_EQ(probe.waits, 0u);
  }
}

TEST(ArGlassesSourceRefresh, MaterialCadenceChangeStillSetsAndVerifiesTheRequestedMode) {
  for (const auto rates : std::array {
         std::array {59940, 60000},
         std::array {60000, 59940},
         std::array {119880, 120000},
         std::array {120000, 119880},
         std::array {60000, 120000},
         std::array {120000, 120025},
       }) {
    source_refresh_probe_t probe;
    probe.measured = rates[0];
    probe.after_set = rates[1];
    EXPECT_EQ(probe.configure(rates[1]), rates[1]);
    EXPECT_EQ(probe.setters, 1u);
    EXPECT_EQ(probe.queries, 2u);
    EXPECT_EQ(probe.waits, 1u);
    EXPECT_EQ(probe.elapsed, 50ms);
  }
}

TEST(ArGlassesSourceRefresh, PostSetClockDifferenceCompletesOnTheFirstVerifiedReadback) {
  source_refresh_probe_t probe;
  probe.measured = 60000;
  probe.after_set = 120000;
  EXPECT_EQ(probe.configure(120013), 120000);
  EXPECT_EQ(probe.setters, 1u);
  EXPECT_EQ(probe.waits, 1u);
  EXPECT_EQ(probe.elapsed, 50ms);
}

TEST(ArGlassesSourceRefresh, UnsupportedRateRetainsMeasuredCadenceAfterBoundedRetries) {
  source_refresh_probe_t probe;
  probe.measured = 60000;
  EXPECT_EQ(probe.configure(120000), 60000);
  EXPECT_EQ(probe.measured, 60000);
  EXPECT_EQ(probe.setters, 3u);
  EXPECT_EQ(probe.waits, 12u);
  EXPECT_EQ(probe.elapsed, 600ms);
}

TEST(ArGlassesSourceRefresh, MissingOrNonpositiveReadbackCannotCertifyAUsableSource) {
  for (const auto measured : {std::optional<int> {}, std::optional<int> {0}, std::optional<int> {-1}}) {
    source_refresh_probe_t probe;
    probe.measured = measured;
    EXPECT_FALSE(probe.configure(120013));
    EXPECT_EQ(probe.waits, 12u);
    EXPECT_EQ(probe.setters, measured ? 3u : 0u);
  }
}

TEST(ArGlassesSourceRefresh, CancellationCannotReportSuccessEvenWhenTheSetterReachedTolerance) {
  source_refresh_probe_t probe;
  probe.cancelled = true;
  EXPECT_FALSE(probe.configure(120013));
  EXPECT_EQ(probe.queries, 0u);
  EXPECT_EQ(probe.setters, 0u);
  EXPECT_EQ(probe.waits, 0u);

  probe.cancelled = false;
  probe.cancel_on_wait = true;
  probe.measured = 60000;
  probe.after_set = 120000;
  EXPECT_FALSE(probe.configure(120013));
  EXPECT_EQ(probe.setters, 1u);
  EXPECT_EQ(probe.waits, 1u);
}

TEST(ArGlassesDiscovery, RecognizesSpecificModelsAndNames) {
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:TCL03D4", "Generic Monitor"));
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:ABC1234", "XREAL Air 2 Pro"));
  EXPECT_TRUE(ar_glasses::is_recognized_ar_display("DISPLAY:ABC1234", "SmartGlasses"));
}

TEST(ArGlassesDiscovery, DoesNotGuessFromOrdinaryMonitorNames) {
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:SMKD1CE", "Apollo AR Des"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:GSM1234", "LG ULTRAGEAR"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:AUS4321", "ROG PG32UCDM"));
  EXPECT_FALSE(ar_glasses::is_recognized_ar_display("DISPLAY:ACI9999", "ARZOPA Portable Monitor"));
}

TEST(ArGlassesRemotePreservation, PreservesApprovedAndRecognizedArDisplays) {
  const std::vector<ar_glasses::device_info_t> decisions {
    {.id = "DISPLAY:USR1234", .name = "User selected display", .decision = ar_glasses::device_decision_e::approved},
  };

  EXPECT_TRUE(ar_glasses::detail::preserve_during_remote_virtual_display_for_test("DISPLAY:USR1234", "Generic Monitor", decisions));
  EXPECT_TRUE(ar_glasses::detail::preserve_during_remote_virtual_display_for_test(
    "DISPLAY:TCL03D4",
    "Generic Monitor",
    {}
  ));
}

TEST(ArGlassesRemotePreservation, ExplicitRejectionOverridesRecognition) {
  const std::vector<ar_glasses::device_info_t> decisions {
    {.id = "display:tcl03d4", .name = "SmartGlasses", .decision = ar_glasses::device_decision_e::rejected},
    {.id = "DISPLAY:TCL03D4", .name = "SmartGlasses", .decision = ar_glasses::device_decision_e::approved},
  };

  EXPECT_FALSE(ar_glasses::detail::preserve_during_remote_virtual_display_for_test(
    "DISPLAY:TCL03D4",
    "SmartGlasses",
    decisions
  ));
}

TEST(ArGlassesRemotePreservation, FailsClosedForAmbiguousOrOrdinaryDisplays) {
  const std::vector<ar_glasses::device_info_t> ambiguous_approval {
    {.id = "DISPLAY:0:0", .name = "SmartGlasses", .decision = ar_glasses::device_decision_e::approved},
  };

  EXPECT_FALSE(ar_glasses::detail::preserve_during_remote_virtual_display_for_test(
    "DISPLAY:0:0",
    "SmartGlasses",
    ambiguous_approval
  ));
  EXPECT_FALSE(ar_glasses::detail::preserve_during_remote_virtual_display_for_test(
    "DISPLAY:GSM9E9D",
    "LG ULTRAGEAR+",
    {}
  ));
}

TEST(ArGlassesOwnership, RenewedRemoteConnectWindowBlocksLocalPresentation) {
  constexpr ar_glasses::remote_virtual_display_lease_t lease = 1001;
  ar_glasses::remote_virtual_display_ended(lease);
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(lease, 0ms));

  // This is called after display creation, encoder probing, and app preparation. Its fresh lease
  // must remain visible to the local topology controller until RTSP activates or the lease ends.
  ar_glasses::remote_virtual_display_awaiting_client(lease, 0ms);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());

  ar_glasses::remote_virtual_display_ended(lease);
  EXPECT_FALSE(ar_glasses::remote_virtual_display_blocks_local());
}

TEST(ArGlassesSessionRequest, ARequestBlockedByAnActiveRemoteCannotLaterStealItsReconnectGrace) {
  ar_glasses::detail::local_session_request_t request;
  request.observe(true);
  ASSERT_TRUE(request.fresh());
  request.consume();
  for (int poll = 0; poll < 100; ++poll) {
    request.observe(true);
    EXPECT_FALSE(request.fresh());
  }
  request.observe(false);
  request.observe(true);
  EXPECT_TRUE(request.fresh());
}

TEST(ArGlassesSessionRequest, ANewPhysicalConnectionGetsOneReplacementRequest) {
  ar_glasses::detail::local_session_request_t request;
  request.observe(true);
  request.consume();
  request.reset();
  request.observe(true);
  EXPECT_TRUE(request.fresh());
  request.observe(false);
  EXPECT_FALSE(request.fresh());
}

TEST(ArGlassesSessionRequest, TransientAdmissionRetriesPreserveTheSameFreshRequest) {
  ar_glasses::detail::local_session_request_t request;
  request.observe(true);
  request.observe(true);
  EXPECT_TRUE(request.fresh());
  request.consume();
  EXPECT_FALSE(request.fresh());
}

TEST(ArGlassesSessionRequest, AShortSamePathUsbReconnectCreatesOnlyOneFreshRequest) {
  ar_glasses::detail::local_session_request_t request;
  request.observe_connection(true, true);
  request.observe(true);
  request.consume();
  for (int poll = 0; poll < 3; ++poll) {
    request.observe_connection(false, false);
    request.observe(true);  // Last confirmed worn state survives the brief sensor gap.
    EXPECT_FALSE(request.fresh());
  }
  request.observe_connection(true, false);
  request.observe(true);
  ASSERT_TRUE(request.fresh());
  request.consume();
  request.observe_connection(true, true);
  request.observe(true);
  EXPECT_FALSE(request.fresh());
}

TEST(ArGlassesSessionRequest, ALiveUsbSubscriptionKeepsADpModeGapFromBecomingANewConnection) {
  ar_glasses::detail::local_session_request_t request;
  request.observe(true);
  request.consume();
  request.observe_connection(false, true);
  request.observe(true);
  request.observe_connection(true, true);
  request.observe(true);
  EXPECT_FALSE(request.fresh());
}

TEST(ArGlassesSessionRequest, SensorRecoveryWithoutDisplayDisconnectDoesNotCreateAFreshConnection) {
  ar_glasses::detail::local_session_request_t request;
  request.observe(true);
  request.consume();
  request.observe_connection(true, false);
  request.observe(true);
  request.observe_connection(true, true);
  request.observe(true);
  EXPECT_FALSE(request.fresh());
}

class ArGlassesWearActivation: public testing::Test {
protected:
  using policy_t = ar_glasses::detail::local_session_activation_t;
  using activation_e = ar_glasses::detail::local_session_activation_e;
  using availability_e = ar_glasses::rayneo::wear_availability_e;
  using state_e = ar_glasses::rayneo::wear_state_e;
  using snapshot_t = ar_glasses::rayneo::wear_snapshot_t;

  static constexpr policy_t::clock_t::time_point origin {};
};

TEST_F(ArGlassesWearActivation, ADisplayWithoutAWearProtocolUsesItsConnectionImmediately) {
  policy_t policy;
  EXPECT_EQ(policy.observe(std::nullopt, origin), activation_e::connection);
  EXPECT_EQ(policy.observe(std::nullopt, origin + 1h), activation_e::connection);
  EXPECT_TRUE(ar_glasses::detail::local_session_requested(activation_e::connection));
}

TEST_F(ArGlassesWearActivation, InitialUnknownSensorStatesWaitBeforeConnectionFallback) {
  for (const auto availability : {availability_e::probing, availability_e::available, availability_e::unavailable}) {
    SCOPED_TRACE(static_cast<int>(availability));
    policy_t policy;
    const snapshot_t snapshot {availability, state_e::unknown};
    EXPECT_EQ(policy.observe(snapshot, origin), activation_e::waiting_for_sensor);
    EXPECT_EQ(policy.observe(snapshot, origin + 2999ms), activation_e::waiting_for_sensor);
    EXPECT_EQ(policy.observe(snapshot, origin + 3s), activation_e::connection);
    EXPECT_EQ(policy.observe(snapshot, origin + 1h), activation_e::connection);
  }
  EXPECT_FALSE(ar_glasses::detail::local_session_requested(activation_e::waiting_for_sensor));
}

TEST_F(ArGlassesWearActivation, AnInitialOffHeadReportNeverStartsTheVirtualDesktop) {
  policy_t policy;
  const snapshot_t off_head {availability_e::available, state_e::off_head};
  EXPECT_EQ(policy.observe(off_head, origin), activation_e::off_head);
  EXPECT_EQ(policy.observe(off_head, origin + 1h), activation_e::off_head);
  EXPECT_FALSE(ar_glasses::detail::local_session_requested(activation_e::off_head));
}

TEST_F(ArGlassesWearActivation, WornAndOffHeadControlTheWholeSessionWithoutASecondGracePeriod) {
  policy_t policy;
  const snapshot_t worn {availability_e::available, state_e::worn};
  const snapshot_t off_head {availability_e::available, state_e::off_head};
  EXPECT_EQ(policy.observe(worn, origin), activation_e::worn);
  EXPECT_EQ(policy.observe(off_head, origin + 250ms), activation_e::off_head);
  EXPECT_EQ(policy.observe(worn, origin + 500ms), activation_e::worn);
  EXPECT_TRUE(ar_glasses::detail::local_session_requested(activation_e::worn));
}

TEST_F(ArGlassesWearActivation, AmbiguousReportsAndTransportRecoveryRetainTheLastConfirmedDecision) {
  for (const auto confirmed : {state_e::off_head, state_e::worn}) {
    SCOPED_TRACE(static_cast<int>(confirmed));
    policy_t policy;
    const auto expected = confirmed == state_e::worn ? activation_e::worn : activation_e::off_head;
    ASSERT_EQ(policy.observe(snapshot_t {availability_e::available, confirmed}, origin), expected);

    // These states occur both during a normal proximity transition and during HID removal,
    // stale-subscription recovery, or reauthentication while DisplayPort remains connected.
    EXPECT_EQ(policy.observe(snapshot_t {availability_e::available, state_e::unknown}, origin + 1s), expected);
    EXPECT_EQ(policy.observe(snapshot_t {availability_e::unavailable, state_e::unknown}, origin + 10s), expected);
    EXPECT_EQ(policy.observe(snapshot_t {availability_e::probing, state_e::unknown}, origin + 30s), expected);
    EXPECT_EQ(policy.observe(snapshot_t {availability_e::unavailable, state_e::unknown}, origin + 1h), expected);
  }
}

TEST_F(ArGlassesWearActivation, ARecoveredSensorTakesAuthorityFromConnectionFallback) {
  policy_t policy;
  const snapshot_t unavailable {availability_e::unavailable, state_e::unknown};
  ASSERT_EQ(policy.observe(unavailable, origin), activation_e::waiting_for_sensor);
  ASSERT_EQ(policy.observe(unavailable, origin + 3s), activation_e::connection);
  EXPECT_EQ(policy.observe(snapshot_t {availability_e::available, state_e::off_head}, origin + 4s), activation_e::off_head);
  EXPECT_EQ(policy.observe(snapshot_t {availability_e::available, state_e::worn}, origin + 5s), activation_e::worn);
}

TEST_F(ArGlassesWearActivation, OnlyAnAvailableSensorMayChangeTheConfirmedDecision) {
  policy_t policy;
  ASSERT_EQ(policy.observe(snapshot_t {availability_e::available, state_e::off_head}, origin), activation_e::off_head);
  EXPECT_EQ(policy.observe(snapshot_t {availability_e::probing, state_e::worn}, origin + 1s), activation_e::off_head);
  EXPECT_EQ(policy.observe(snapshot_t {availability_e::unavailable, state_e::worn}, origin + 2s), activation_e::off_head);
  EXPECT_EQ(policy.observe(snapshot_t {availability_e::available, state_e::worn}, origin + 3s), activation_e::worn);
}

TEST_F(ArGlassesWearActivation, DisconnectOrTargetReplacementResetsWearAuthorityAndProbeDeadline) {
  for (const auto previous : {state_e::off_head, state_e::worn}) {
    SCOPED_TRACE(static_cast<int>(previous));
    policy_t policy;
    policy.observe(snapshot_t {availability_e::available, previous}, origin);

    // The controller resets this policy when the selected physical connection disappears or
    // changes identity; an unrelated/reconnected device must not inherit the old sensor state.
    policy = {};
    const auto reconnected_at = origin + 1h;
    const snapshot_t probing {availability_e::probing, state_e::unknown};
    EXPECT_EQ(policy.observe(probing, reconnected_at), activation_e::waiting_for_sensor);
    EXPECT_EQ(policy.observe(probing, reconnected_at + 2999ms), activation_e::waiting_for_sensor);
    EXPECT_EQ(policy.observe(probing, reconnected_at + 3s), activation_e::connection);
  }
}

TEST_F(ArGlassesWearActivation, AConfirmedWearChangeSurvivesUnknownBeforeTheControllerReturns) {
  for (const auto previous : {state_e::off_head, state_e::worn}) {
    SCOPED_TRACE(static_cast<int>(previous));
    const auto changed = previous == state_e::worn ? state_e::off_head : state_e::worn;
    const auto expected = changed == state_e::worn ? activation_e::worn : activation_e::off_head;
    policy_t policy;
    policy.observe(snapshot_t {availability_e::available, previous}, origin);

    // While Windows blocks the controller in setup/teardown, the independent HID worker can
    // confirm the opposite state and then lose its live sample before the next controller poll.
    const std::array publications {
      snapshot_t {availability_e::available, previous},
      snapshot_t {availability_e::available, changed},
      snapshot_t {availability_e::available, state_e::unknown},
      snapshot_t {availability_e::probing, state_e::unknown},
      snapshot_t {availability_e::unavailable, state_e::unknown},
    };
    const auto remembered = ar_glasses::rayneo::detail::last_confirmed_state_for_test(publications);
    ASSERT_EQ(remembered, changed);
    EXPECT_EQ(policy.observe(publications.back(), origin + 5s, remembered), expected);
  }
}

TEST_F(ArGlassesWearActivation, NewerRememberedAuthorityOverridesAnOlderLiveSnapshot) {
  policy_t policy;
  const snapshot_t older_snapshot {availability_e::available, state_e::worn};
  const std::array publications {
    older_snapshot,
    snapshot_t {availability_e::available, state_e::off_head},
  };
  // The worker may publish off-head between the controller's snapshot and history reads.
  const auto remembered = ar_glasses::rayneo::detail::last_confirmed_state_for_test(publications);
  EXPECT_EQ(policy.observe(older_snapshot, origin, remembered), activation_e::off_head);
}

TEST_F(ArGlassesWearActivation, UnknownOrUnauthenticatedPublicationsNeverCreateWearAuthority) {
  const std::array publications {
    snapshot_t {availability_e::probing, state_e::unknown},
    snapshot_t {availability_e::available, state_e::unknown},
    snapshot_t {availability_e::unavailable, state_e::unknown},
    snapshot_t {availability_e::probing, state_e::worn},
    snapshot_t {availability_e::unavailable, state_e::off_head},
  };
  const auto remembered = ar_glasses::rayneo::detail::last_confirmed_state_for_test(publications);
  ASSERT_EQ(remembered, state_e::unknown);
  policy_t policy;
  EXPECT_EQ(policy.observe(publications.back(), origin, remembered), activation_e::waiting_for_sensor);
  EXPECT_EQ(policy.observe(publications.back(), origin + 3s, remembered), activation_e::connection);
  EXPECT_EQ(ar_glasses::rayneo::detail::last_confirmed_state_for_test({}), state_e::unknown);
}

TEST(RayNeoWearReport, DecodesEmpiricallySeparatedWornAndOffHeadBands) {
  const auto worn = ar_glasses::rayneo::detail::parse_sensor_report_for_test(
    rayneo_sensor_report(65535.0f, 100)
  );
  ASSERT_TRUE(worn);
  EXPECT_EQ(worn->tick, 100u);
  EXPECT_FLOAT_EQ(worn->proximity, 65535.0f);
  EXPECT_EQ(worn->state, ar_glasses::rayneo::wear_state_e::worn);

  const auto live_lower_worn = ar_glasses::rayneo::detail::parse_sensor_report_for_test(
    rayneo_sensor_report(22835.0f, 101)
  );
  ASSERT_TRUE(live_lower_worn);
  EXPECT_EQ(live_lower_worn->state, ar_glasses::rayneo::wear_state_e::worn);

  const auto off_head = ar_glasses::rayneo::detail::parse_sensor_report_for_test(
    rayneo_sensor_report(1545.0f, 102)
  );
  ASSERT_TRUE(off_head);
  EXPECT_EQ(off_head->state, ar_glasses::rayneo::wear_state_e::off_head);

  const auto intermediate = ar_glasses::rayneo::detail::parse_sensor_report_for_test(
    rayneo_sensor_report(12500.0f, 103)
  );
  ASSERT_TRUE(intermediate);
  EXPECT_EQ(intermediate->state, ar_glasses::rayneo::wear_state_e::unknown);

  EXPECT_EQ(
    ar_glasses::rayneo::detail::parse_sensor_report_for_test(rayneo_sensor_report(10000.0f))->state,
    ar_glasses::rayneo::wear_state_e::off_head
  );
  EXPECT_EQ(
    ar_glasses::rayneo::detail::parse_sensor_report_for_test(rayneo_sensor_report(10001.0f))->state,
    ar_glasses::rayneo::wear_state_e::unknown
  );
  EXPECT_EQ(
    ar_glasses::rayneo::detail::parse_sensor_report_for_test(rayneo_sensor_report(14999.0f))->state,
    ar_glasses::rayneo::wear_state_e::unknown
  );
  EXPECT_EQ(
    ar_glasses::rayneo::detail::parse_sensor_report_for_test(rayneo_sensor_report(15000.0f))->state,
    ar_glasses::rayneo::wear_state_e::worn
  );
}

TEST(RayNeoWearReport, RejectsWrongFramingAndLength) {
  auto ack = rayneo_sensor_report(65535.0f);
  ack[2] = 0xC8;
  EXPECT_FALSE(ar_glasses::rayneo::detail::parse_sensor_report_for_test(ack));

  auto short_report = rayneo_sensor_report(65535.0f);
  EXPECT_FALSE(ar_glasses::rayneo::detail::parse_sensor_report_for_test(std::span<const std::uint8_t>(short_report).first(64)));
}

TEST(RayNeoWearReport, MapsImplausibleProximityToUnknown) {
  for (const auto proximity : {
         std::numeric_limits<float>::quiet_NaN(),
         std::numeric_limits<float>::infinity(),
         -1.0f,
         65536.0f,
       }) {
    const auto parsed = ar_glasses::rayneo::detail::parse_sensor_report_for_test(
      rayneo_sensor_report(proximity)
    );
    ASSERT_TRUE(parsed);
    EXPECT_EQ(parsed->state, ar_glasses::rayneo::wear_state_e::unknown);
  }
}

TEST(RayNeoWearReport, ClassificationDoesNotDependOnImuOrLightBytes) {
  const auto zero_filled = ar_glasses::rayneo::detail::parse_sensor_report_for_test(
    rayneo_sensor_report(65535.0f, 200, 0x00)
  );
  const auto changed_motion = ar_glasses::rayneo::detail::parse_sensor_report_for_test(
    rayneo_sensor_report(65535.0f, 201, 0xA5)
  );
  ASSERT_TRUE(zero_filled);
  ASSERT_TRUE(changed_motion);
  EXPECT_EQ(zero_filled->state, changed_motion->state);
  EXPECT_EQ(changed_motion->state, ar_glasses::rayneo::wear_state_e::worn);
}

TEST(RayNeoWearIdentity, RequiresTheAuthenticatedAir4ProBoard) {
  EXPECT_TRUE(ar_glasses::rayneo::detail::board_info_matches_for_test(rayneo_board_info_ack(0x3A)));
  EXPECT_FALSE(ar_glasses::rayneo::detail::board_info_matches_for_test(rayneo_board_info_ack(0x39)));

  auto wrong_command = rayneo_board_info_ack(0x3A);
  wrong_command[1 + 8] = 0x01;
  EXPECT_FALSE(ar_glasses::rayneo::detail::board_info_matches_for_test(wrong_command));
  EXPECT_FALSE(ar_glasses::rayneo::detail::board_info_matches_for_test(rayneo_sensor_report(65535.0f)));
}

TEST(RayNeoWearIdentity, DoesNotStartAHidMonitorForOtherDisplays) {
  EXPECT_FALSE(ar_glasses::rayneo::create_wear_monitor("DISPLAY:ABC1234"));
  EXPECT_FALSE(ar_glasses::rayneo::create_wear_monitor("display:tcl03d4"));
}

TEST(RayNeoWearRecovery, FiltersLifecycleNotificationsToTheExactUsbIdentity) {
  EXPECT_TRUE(ar_glasses::rayneo::detail::hid_interface_matches_for_test(LR"(\\?\HID#VID_1BBB&PID_AF50&MI_00&COL01#test)"));
  EXPECT_TRUE(ar_glasses::rayneo::detail::hid_interface_matches_for_test(LR"(\\?\hid#vid_1bbb&pid_af50#test)"));
  EXPECT_FALSE(ar_glasses::rayneo::detail::hid_interface_matches_for_test(LR"(\\?\hid#vid_1bbb&pid_af51#test)"));
  EXPECT_FALSE(ar_glasses::rayneo::detail::hid_interface_matches_for_test(LR"(\\?\hid#xvid_1bbb&pid_af50#test)"));
  EXPECT_FALSE(ar_glasses::rayneo::detail::hid_interface_matches_for_test(LR"(\\?\hid#vid_1bbb&pid_af500#test)"));
  EXPECT_FALSE(ar_glasses::rayneo::detail::hid_interface_matches_for_test(LR"(\\?\hid#vid_3941&pid_1a04#test)"));
}

TEST(RayNeoWearRecovery, ClassifiesEveryWindowsSuspendResumeVariant) {
  using power_event_e = ar_glasses::rayneo::detail::power_event_e;
  const auto classify = ar_glasses::rayneo::detail::power_event_for_test;

  EXPECT_EQ(classify(PBT_APMSUSPEND), power_event_e::suspend);
  EXPECT_EQ(classify(PBT_APMRESUMEAUTOMATIC), power_event_e::resume);
  EXPECT_EQ(classify(PBT_APMRESUMECRITICAL), power_event_e::resume);
  EXPECT_EQ(classify(PBT_APMRESUMESUSPEND), power_event_e::resume);
  EXPECT_EQ(classify(PBT_APMPOWERSTATUSCHANGE), power_event_e::none);
}

TEST(RayNeoWearRecovery, CoalescesThePairedWindowsResumeCallbacks) {
  const auto requests_recovery = ar_glasses::rayneo::detail::power_event_requests_recovery_for_test;

  EXPECT_TRUE(requests_recovery(PBT_APMSUSPEND, false));
  EXPECT_TRUE(requests_recovery(PBT_APMRESUMEAUTOMATIC, false));
  EXPECT_TRUE(requests_recovery(PBT_APMRESUMECRITICAL, false));
  EXPECT_TRUE(requests_recovery(PBT_APMRESUMESUSPEND, true));
  EXPECT_FALSE(requests_recovery(PBT_APMRESUMESUSPEND, false));
}

TEST(RayNeoWearRecovery, UsesEventsWithABoundedSafetyFallback) {
  const auto retry_delay = ar_glasses::rayneo::detail::recovery_retry_delay_for_test;

  EXPECT_EQ(retry_delay(true, true, 0), 30s);
  EXPECT_EQ(retry_delay(true, true, 100), 30s);
  EXPECT_EQ(retry_delay(true, false, 0), 1s);
  EXPECT_EQ(retry_delay(false, false, 100), 1s);

  EXPECT_EQ(retry_delay(false, true, 0), 1s);
  EXPECT_EQ(retry_delay(false, true, 1), 2s);
  EXPECT_EQ(retry_delay(false, true, 2), 5s);
  EXPECT_EQ(retry_delay(false, true, 3), 10s);
  EXPECT_EQ(retry_delay(false, true, 4), 30s);
  EXPECT_EQ(retry_delay(false, true, 100), 30s);
}

TEST(RayNeoWearDebounce, RequiresContinuousEvidenceAndExpiresStaleStream) {
  using observation_t = ar_glasses::rayneo::detail::debounce_observation_t;
  using state_e = ar_glasses::rayneo::wear_state_e;
  const std::array off_head_samples {
    observation_t {0ms, 1, state_e::off_head},
    observation_t {75ms, 2, state_e::off_head},
    observation_t {149ms, 3, state_e::off_head},
  };
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(off_head_samples, 149ms),
    state_e::unknown
  );

  const std::array confirmed_off_head {
    observation_t {0ms, 1, state_e::off_head},
    observation_t {75ms, 2, state_e::off_head},
    observation_t {150ms, 3, state_e::off_head},
  };
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(confirmed_off_head, 150ms),
    state_e::off_head
  );
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(confirmed_off_head, 10149ms),
    state_e::off_head
  );
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(confirmed_off_head, 10150ms),
    state_e::unknown
  );
}

TEST(RayNeoWearDebounce, IrregularDeviceTicksStillConfirmAndRefreshWornState) {
  using observation_t = ar_glasses::rayneo::detail::debounce_observation_t;
  using state_e = ar_glasses::rayneo::wear_state_e;
  const std::array irregular_tick_samples {
    observation_t {0ms, 100, state_e::worn},
    observation_t {75ms, 100, state_e::worn},
    observation_t {150ms, 90, state_e::worn},
    observation_t {400ms, 0, state_e::worn},
  };

  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(irregular_tick_samples, 10399ms),
    state_e::worn
  );
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(irregular_tick_samples, 10400ms),
    state_e::unknown
  );

  const std::array repeated_tick_off_head {
    observation_t {0ms, 7, state_e::off_head},
    observation_t {150ms, 7, state_e::off_head},
    observation_t {400ms, 7, state_e::off_head},
  };
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(repeated_tick_off_head, 10399ms),
    state_e::off_head
  );
}

TEST(RayNeoWearDebounce, LiveWornBandsRemainOneContinuousState) {
  using observation_t = ar_glasses::rayneo::detail::debounce_observation_t;
  using state_e = ar_glasses::rayneo::wear_state_e;
  const std::array observations {
    observation_t {0ms, 1, state_e::worn},
    observation_t {150ms, 2, state_e::worn},
    observation_t {500ms, 3, state_e::worn},
    observation_t {1000ms, 4, state_e::worn},
    observation_t {1500ms, 5, state_e::worn},
  };

  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(observations, 1500ms),
    state_e::worn
  );
}

TEST(RayNeoWearDebounce, AStaleGapRequiresAFreshContinuousProposal) {
  using observation_t = ar_glasses::rayneo::detail::debounce_observation_t;
  using state_e = ar_glasses::rayneo::wear_state_e;
  const std::array before_confirmation {
    observation_t {0ms, 1, state_e::worn},
    observation_t {150ms, 2, state_e::worn},
    observation_t {10200ms, 3, state_e::off_head},
    observation_t {10349ms, 4, state_e::off_head},
  };
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(before_confirmation, 10349ms),
    state_e::unknown
  );

  const std::array after_confirmation {
    observation_t {0ms, 1, state_e::worn},
    observation_t {150ms, 2, state_e::worn},
    observation_t {10200ms, 3, state_e::off_head},
    observation_t {10350ms, 4, state_e::off_head},
  };
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(after_confirmation, 10350ms),
    state_e::off_head
  );
}

TEST(RayNeoWearDebounce, AGlitchRestartsTheOppositeStateWindow) {
  using observation_t = ar_glasses::rayneo::detail::debounce_observation_t;
  using state_e = ar_glasses::rayneo::wear_state_e;
  const std::array before_confirmation {
    observation_t {0ms, 1, state_e::off_head},
    observation_t {100ms, 2, state_e::off_head},
    observation_t {120ms, 3, state_e::unknown},
    observation_t {200ms, 4, state_e::off_head},
    observation_t {349ms, 5, state_e::off_head},
  };
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(before_confirmation, 349ms),
    state_e::unknown
  );

  const std::array after_confirmation {
    observation_t {0ms, 1, state_e::off_head},
    observation_t {100ms, 2, state_e::off_head},
    observation_t {120ms, 3, state_e::unknown},
    observation_t {200ms, 4, state_e::off_head},
    observation_t {350ms, 5, state_e::off_head},
  };
  EXPECT_EQ(
    ar_glasses::rayneo::detail::debounce_observations_for_test(after_confirmation, 350ms),
    state_e::off_head
  );
}

TEST(ArGlassesOwnership, ProcessSetupPinsLeasePastTheInitialConnectWindow) {
  constexpr ar_glasses::remote_virtual_display_lease_t lease = 1051;
  ar_glasses::remote_virtual_display_ended(lease);
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(lease, 0ms, true));

  // A zero connect timeout still has the production two-second handshake floor. Process setup can
  // legitimately exceed it while probing encoders or preparing an app, so the setup pin must keep
  // the lease alive until the post-setup renewal starts the ordinary bounded client window.
  std::this_thread::sleep_for(2200ms);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());
  ar_glasses::remote_virtual_display_awaiting_client(lease, 0ms);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_active(lease));

  ar_glasses::remote_virtual_display_ended(lease);
  EXPECT_FALSE(ar_glasses::remote_virtual_display_blocks_local());
}

TEST(ArGlassesOwnership, OlderLifecycleCannotClearNewerReconnectLease) {
  constexpr ar_glasses::remote_virtual_display_lease_t old_lease = 1101;
  constexpr ar_glasses::remote_virtual_display_lease_t reconnect_lease = 1102;
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(old_lease, 0ms));
  ASSERT_TRUE(ar_glasses::remote_virtual_display_starting(reconnect_lease, 0ms));

  ar_glasses::remote_virtual_display_ended(old_lease);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());
  EXPECT_TRUE(ar_glasses::remote_virtual_display_active(reconnect_lease));

  ar_glasses::remote_virtual_display_ended(old_lease);
  EXPECT_TRUE(ar_glasses::remote_virtual_display_blocks_local());
  ar_glasses::remote_virtual_display_ended(reconnect_lease);
  EXPECT_FALSE(ar_glasses::remote_virtual_display_blocks_local());
}

TEST(ArGlassesOwnership, ReservationTracksLongConfiguredPingTimeout) {
  EXPECT_EQ(
    ar_glasses::detail::remote_pending_duration(120s),
    122s
  );
  EXPECT_EQ(
    ar_glasses::detail::remote_pending_duration(-1ms),
    2s
  );
}

TEST(ArGlassesAdapterContract, TracksThePhysicalOutputAcrossTopologyChanges) {
  const LUID physical_before {.LowPart = 0x12345678u, .HighPart = 0x1234};
  const LUID same_physical_after {.LowPart = 0x12345678u, .HighPart = 0x1234};
  const LUID different_virtual_adapter {.LowPart = 0x87654321u, .HighPart = 0x4321};

  // The virtual target may identify SudoVDA rather than its render GPU. Only the stable physical
  // target before/after topology mutation is authoritative.
  EXPECT_TRUE(ar_glasses::detail::physical_adapter_contract_valid_for_test(
    physical_before,
    same_physical_after,
    different_virtual_adapter
  ));
}

TEST(ArGlassesAdapterContract, RejectsAPhysicalOutputThatMigratedAdapters) {
  const LUID physical_before {.LowPart = 0x12345678u, .HighPart = 0x1234};
  const LUID physical_after {.LowPart = 0x87654321u, .HighPart = 0x4321};
  const LUID virtual_adapter = physical_after;

  // Matching the new physical adapter to SudoVDA must not hide a real physical-output migration.
  EXPECT_FALSE(ar_glasses::detail::physical_adapter_contract_valid_for_test(
    physical_before,
    physical_after,
    virtual_adapter
  ));
}

TEST(ArGlassesLinearLayout, PlacesSourceThenSinkAfterTheRightmostInteractiveOutput) {
  const RECT anchor {-2560, 300, 1280, 1740};

  const auto layout = ar_glasses::detail::compute_linear_layout_for_test(
    anchor,
    1920,
    1080,
    3840,
    1080
  );

  EXPECT_EQ(layout.virtual_rect.left, 1280);
  EXPECT_EQ(layout.virtual_rect.top, 300);
  EXPECT_EQ(layout.virtual_rect.right, 3200);
  EXPECT_EQ(layout.virtual_rect.bottom, 1380);
  EXPECT_EQ(layout.physical_rect.left, 3200);
  EXPECT_EQ(layout.physical_rect.top, 300);
  EXPECT_EQ(layout.physical_rect.right, 7040);
  EXPECT_EQ(layout.physical_rect.bottom, 1380);
}

TEST(ArGlassesLinearLayout, DependsOnTheAnchorRatherThanAbsoluteDesktopCoordinates) {
  const RECT first_anchor {0, 0, 2560, 1440};
  const RECT shifted_anchor {-4000, -700, -1440, 740};

  const auto first = ar_glasses::detail::compute_linear_layout_for_test(
    first_anchor,
    1920,
    1080,
    1920,
    1080
  );
  const auto shifted = ar_glasses::detail::compute_linear_layout_for_test(
    shifted_anchor,
    1920,
    1080,
    1920,
    1080
  );

  EXPECT_EQ(first.virtual_rect.left, first_anchor.right);
  EXPECT_EQ(first.physical_rect.left, first.virtual_rect.right);
  EXPECT_EQ(shifted.virtual_rect.left, shifted_anchor.right);
  EXPECT_EQ(shifted.physical_rect.left, shifted.virtual_rect.right);
  EXPECT_EQ(shifted.virtual_rect.top, shifted_anchor.top);
  EXPECT_EQ(shifted.physical_rect.top, shifted_anchor.top);
}

TEST(ArGlassesLinearLayout, SelectsAnchorDeterministicallyWhenRightEdgesTie) {
  const ar_glasses::detail::anchor_candidate_t upper {
    .rect = {0, -1080, 1920, 0},
    .source_adapter_id = {.LowPart = 2, .HighPart = 0},
    .source_id = 2,
  };
  const ar_glasses::detail::anchor_candidate_t lower {
    .rect = {0, 0, 1920, 1080},
    .source_adapter_id = {.LowPart = 1, .HighPart = 0},
    .source_id = 1,
  };

  const auto forward = ar_glasses::detail::select_anchor_for_test({lower, upper});
  const auto reversed = ar_glasses::detail::select_anchor_for_test({upper, lower});
  ASSERT_TRUE(forward);
  ASSERT_TRUE(reversed);
  EXPECT_EQ(forward->rect.top, -1080);
  EXPECT_EQ(reversed->rect.top, -1080);
}

TEST(ArGlassesLinearLayout, RejectsAnyCloneOfThePhysicalSource) {
  const LUID adapter {.LowPart = 7, .HighPart = 0};
  const std::vector<ar_glasses::detail::topology_path_identity_t> paths {
    {adapter, 3, adapter, 10},
    {adapter, 3, adapter, 11},
    {adapter, 4, adapter, 12},
  };

  EXPECT_TRUE(ar_glasses::detail::source_is_cloned_for_test(paths, 0));
  EXPECT_TRUE(ar_glasses::detail::source_is_cloned_for_test(paths, 1));
  EXPECT_FALSE(ar_glasses::detail::source_is_cloned_for_test(paths, 2));
}

TEST(ArGlassesTopologyEvidence, RejectsMissingOrStalePrimarySourceEvidence) {
  const std::vector<std::wstring> active_sources {LR"(\\.\DISPLAY1)", LR"(\\.\DISPLAY4)"};

  EXPECT_TRUE(ar_glasses::detail::primary_source_is_authoritative_for_test(
    LR"(\\.\DISPLAY1)",
    active_sources
  ));
  EXPECT_FALSE(ar_glasses::detail::primary_source_is_authoritative_for_test(L"", active_sources));
  EXPECT_FALSE(ar_glasses::detail::primary_source_is_authoritative_for_test(
    LR"(\\.\DISPLAY2)",
    active_sources
  ));
}

TEST(ArGlassesLinearLayout, RequiresBothExactRectanglesAndSharedEdge) {
  const ar_glasses::detail::linear_layout_t expected {
    .virtual_rect = {1920, 0, 3840, 1080},
    .physical_rect = {3840, 0, 7680, 1080},
  };
  EXPECT_TRUE(ar_glasses::detail::isolated_layout_matches_for_test(
    expected,
    expected.virtual_rect,
    expected.physical_rect
  ));

  RECT normalized_virtual = expected.virtual_rect;
  normalized_virtual.top = 1;
  normalized_virtual.bottom = 1081;
  EXPECT_FALSE(ar_glasses::detail::isolated_layout_matches_for_test(
    expected,
    normalized_virtual,
    expected.physical_rect
  ));
}

TEST(ArGlassesTopologyRecovery, SerializesAPlainRecordArray) {
  auto recoveries = nlohmann::json::array();
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));

  const auto result = ar_glasses::detail::parse_topology_recovery_json_for_test(
    recovery_document(std::move(recoveries), 4).dump()
  );

  ASSERT_TRUE(result.valid);
  EXPECT_EQ(result.record_count, 1u);
  const auto normalized = nlohmann::json::parse(result.normalized_json);
  EXPECT_EQ(normalized.at("version"), 4);
  ASSERT_TRUE(normalized.at("recoveries").is_array());
  ASSERT_EQ(normalized.at("recoveries").size(), 1u);
  EXPECT_TRUE(normalized.at("recoveries").front().is_object());
}

TEST(ArGlassesTopologyRecovery, RejectsRetiredVersionTwoAndThreeRecords) {
  auto version_two = legacy_recovery_record();
  version_two["version"] = 2;

  auto version_three_entries = nlohmann::json::array();
  version_three_entries.emplace_back(legacy_recovery_record());
  const auto version_three = recovery_document(std::move(version_three_entries), 3);

  EXPECT_FALSE(ar_glasses::detail::parse_topology_recovery_json_for_test(version_two.dump()).valid);
  EXPECT_FALSE(ar_glasses::detail::parse_topology_recovery_json_for_test(version_three.dump()).valid);
}

TEST(ArGlassesTopologyRecovery, PendingMoveOwnsOnlyExactPendingAndConfirmedPositions) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  owned.emplace_back(rect_json(7040, -1080, 8960, 0));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(
    std::move(owned),
    rect_json(8960, 0, 10880, 1080)
  ));
  const auto document = recovery_document(std::move(recoveries), 4);

  const auto parsed = ar_glasses::detail::parse_topology_recovery_json_for_test(document.dump());
  ASSERT_TRUE(parsed.valid);
  const auto normalized = nlohmann::json::parse(parsed.normalized_json);
  const auto &record = normalized.at("recoveries").front();
  EXPECT_EQ(record.at("owned_rects").size(), 2u);
  EXPECT_FALSE(record.at("pending_rect").is_null());

  const RECT exact_pending {8960, 0, 10880, 1080};
  EXPECT_TRUE(ar_glasses::detail::topology_recovery_should_restore_for_test(
    document.dump(),
    exact_pending
  ));
  const RECT prior_confirmed {7040, -1080, 8960, 0};
  EXPECT_TRUE(ar_glasses::detail::topology_recovery_should_restore_for_test(
    document.dump(),
    prior_confirmed
  ));

  // Apollo may stop after writing the pending marker but before SetDisplayConfig. A different
  // rectangle can therefore be a later user move and must not be claimed by the transaction.
  const RECT unrelated_user_position {8500, 200, 10420, 1280};
  EXPECT_FALSE(ar_glasses::detail::topology_recovery_should_restore_for_test(
    document.dump(),
    unrelated_user_position
  ));
}

TEST(ArGlassesTopologyRecovery, DoesNotOwnAnUnrecognizedRectWithoutAPendingMove) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));
  const auto document = recovery_document(std::move(recoveries), 4);
  const RECT user_position {100, 100, 2020, 1180};

  EXPECT_FALSE(ar_glasses::detail::topology_recovery_should_restore_for_test(
    document.dump(),
    user_position
  ));
}

TEST(ArGlassesTopologyRecovery, RejectsTheRetiredLeadingEmptyArrayArtifact) {
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(nlohmann::json::array());
  recoveries.emplace_back(current_recovery_record());

  const auto result = ar_glasses::detail::parse_topology_recovery_json_for_test(
    recovery_document(std::move(recoveries)).dump()
  );

  EXPECT_FALSE(result.valid);
}

TEST(ArGlassesTopologyRecovery, RejectsOtherMalformedRecoveryLists) {
  auto sentinel_only = nlohmann::json::array();
  sentinel_only.emplace_back(nlohmann::json::array());

  auto trailing_array = nlohmann::json::array();
  trailing_array.emplace_back(current_recovery_record());
  trailing_array.emplace_back(nlohmann::json::array());

  auto repeated_sentinel = nlohmann::json::array();
  repeated_sentinel.emplace_back(nlohmann::json::array());
  repeated_sentinel.emplace_back(nlohmann::json::array());
  repeated_sentinel.emplace_back(current_recovery_record());

  auto null_entry = nlohmann::json::array();
  null_entry.emplace_back(nullptr);
  null_entry.emplace_back(current_recovery_record());

  auto duplicate_devices = nlohmann::json::array();
  duplicate_devices.emplace_back(current_recovery_record());
  duplicate_devices.emplace_back(current_recovery_record());

  auto invalid_rectangle = current_recovery_record();
  invalid_rectangle["owned_rects"][0]["right"] =
    invalid_rectangle["owned_rects"][0]["left"];
  auto malformed_record = nlohmann::json::array();
  malformed_record.emplace_back(std::move(invalid_rectangle));

  for (auto &invalid : {
         std::move(sentinel_only),
         std::move(trailing_array),
         std::move(repeated_sentinel),
         std::move(null_entry),
         std::move(duplicate_devices),
         std::move(malformed_record),
       }) {
    EXPECT_FALSE(ar_glasses::detail::parse_topology_recovery_json_for_test(
                   recovery_document(std::move(invalid)).dump()
                 ).valid);
  }
}

TEST(ArGlassesModeTransition, KeepsVirtualDesktopForSupportedModesOnSameOutput) {
  ar_glasses::detail::local_session_contract_t before;
  before.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  before.adapter_id.LowPart = 42;
  before.adapter_id.HighPart = 7;
  before.mode = ar_glasses::presentation_mode_e::normal;
  before.hdr_known = true;

  auto after = before;
  after.mode = ar_glasses::presentation_mode_e::sbs_ai;

  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after, false));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after, true));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(after, before, false));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(after, before, true));
}

TEST(ArGlassesModeTransition, RebuildsOnlyForOutputAdapterOrUnsupportedChanges) {
  ar_glasses::detail::local_session_contract_t before;
  before.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  before.adapter_id.LowPart = 42;
  before.mode = ar_glasses::presentation_mode_e::normal;
  before.hdr_known = true;

  auto changed = before;
  changed.device_path = LR"(\\?\DISPLAY#OTHER#test)";
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, false));
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, true));

  changed = before;
  changed.adapter_id.LowPart = 43;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, false));
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, true));

  changed = before;
  changed.hdr_active = true;
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, false));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, true));

  changed = before;
  changed.hdr_known = false;
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, false));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, true));

  changed = before;
  changed.mode = ar_glasses::presentation_mode_e::unsupported;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, false));
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, true));

  auto unsupported_before = before;
  unsupported_before.mode = ar_glasses::presentation_mode_e::unsupported;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(unsupported_before, before, false));
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(unsupported_before, before, true));

  changed = before;
  changed.is_primary = true;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, false));

  changed = before;
  changed.is_cloned = true;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, false));
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed, true));
}

TEST(ArGlassesModeTransition, ExclusiveSessionRepairsATemporarilyPrimarySink) {
  ar_glasses::detail::local_session_contract_t before;
  before.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  before.adapter_id.LowPart = 42;
  before.adapter_id.HighPart = 7;
  before.mode = ar_glasses::presentation_mode_e::normal;

  auto after = before;
  after.mode = ar_glasses::presentation_mode_e::sbs_ai;
  after.is_primary = true;

  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after, false));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after, true));

  after.is_cloned = true;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after, true));
}

TEST(ArGlassesModeTransition, PlacementOnlyUpdatesKeepThePresentationContract) {
  ar_glasses::detail::local_session_contract_t expected;
  expected.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  expected.adapter_id.LowPart = 42;
  expected.mode = ar_glasses::presentation_mode_e::normal;
  expected.rect = {1920, 0, 3840, 1080};
  expected.refresh_millihz = 120000;

  auto placed = expected;
  placed.rect = {-1920, 0, 0, 1080};
  EXPECT_TRUE(ar_glasses::detail::local_presenter_target_contract_matches_for_test(expected, placed));

  // Even a repairable primary sink cannot become a presenter target until promotion succeeds.
  placed.is_primary = true;
  EXPECT_FALSE(ar_glasses::detail::local_presenter_target_contract_matches_for_test(expected, placed));
  placed = expected;
  placed.is_cloned = true;
  EXPECT_FALSE(ar_glasses::detail::local_presenter_target_contract_matches_for_test(expected, placed));
}

TEST(ArGlassesModeTransition, PlacementCannotAdoptAConcurrentModeOrColorChange) {
  ar_glasses::detail::local_session_contract_t expected;
  expected.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  expected.adapter_id.LowPart = 42;
  expected.mode = ar_glasses::presentation_mode_e::normal;
  expected.rect = {1920, 0, 3840, 1080};
  expected.refresh_millihz = 120000;

  auto changed = expected;
  changed.mode = ar_glasses::presentation_mode_e::sbs_ai;
  changed.rect.right = changed.rect.left + 3840;
  changed.refresh_millihz = 60000;
  EXPECT_FALSE(ar_glasses::detail::local_presenter_target_contract_matches_for_test(expected, changed));

  changed = expected;
  changed.refresh_millihz = 60000;
  EXPECT_FALSE(ar_glasses::detail::local_presenter_target_contract_matches_for_test(expected, changed));
  changed = expected;
  changed.hdr_active = true;
  EXPECT_FALSE(ar_glasses::detail::local_presenter_target_contract_matches_for_test(expected, changed));
  changed = expected;
  changed.adapter_id.LowPart = 43;
  EXPECT_FALSE(ar_glasses::detail::local_presenter_target_contract_matches_for_test(expected, changed));
  changed = expected;
  changed.device_path = LR"(\\?\DISPLAY#OTHER#test)";
  EXPECT_FALSE(ar_glasses::detail::local_presenter_target_contract_matches_for_test(expected, changed));
}

TEST(ArGlassesModeTransition, FailedPlacementRoutesToRetainedModeAndColorRepair) {
  ar_glasses::detail::local_session_contract_t before;
  before.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  before.adapter_id.LowPart = 42;
  before.mode = ar_glasses::presentation_mode_e::normal;
  before.rect = {1920, 0, 3840, 1080};

  auto moved = before;
  moved.rect = {-1920, 0, 0, 1080};
  EXPECT_TRUE(ar_glasses::detail::local_session_can_re_isolate_for_test(before, moved, false));

  // The sink can remain unchanged when placement discovers mismatched source HDR. Pausing on
  // that failure must select full retained reconfiguration on the next controller retry.
  EXPECT_FALSE(ar_glasses::detail::local_session_can_re_isolate_for_test(before, moved, true));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, moved, false));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, moved, true));
}

TEST(ArGlassesModeTransition, SwitchingTheSourceOnRunningGlassesRebuildsThePresenter) {
  ar_glasses::detail::local_session_contract_t host;
  host.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  host.adapter_id.LowPart = 42;
  host.mode = ar_glasses::presentation_mode_e::normal;
  host.rect = {1920, 0, 3840, 1080};

  // The presenter converts one source. Updating placement alone would keep showing the old
  // one, so a source change on an unchanged sink must take the retained reconfiguration.
  auto game = host;
  game.game_source = true;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_re_isolate_for_test(host, game, false));
  EXPECT_FALSE(ar_glasses::detail::local_session_can_re_isolate_for_test(game, host, false));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(host, game, false));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_re_isolate_for_test(game, game, false));
}

TEST(ArGlassesModeTransition, RetirementMovesTheOwnerAndLatchesCleanupAcrossWaiters) {
  const auto result = ar_glasses::detail::local_retirement_transfer_for_test();
  EXPECT_TRUE(result.transferred);
  EXPECT_TRUE(result.source_released);
  EXPECT_FALSE(result.first_wait_completed);
  EXPECT_TRUE(result.barrier_retained);
  EXPECT_TRUE(result.second_wait_completed);
  EXPECT_TRUE(result.barrier_released);
  EXPECT_EQ(result.prepare_calls, 1);
  EXPECT_EQ(result.remove_calls, 2);
  EXPECT_EQ(result.finish_calls, 1);
}

TEST(ArGlassesModeTransition, PauseRestoresAfterCaptureStopsBeforeWaitingForCleanup) {
  const auto result = ar_glasses::detail::local_pause_order_for_test();
  EXPECT_TRUE(result.pause_completed);
  EXPECT_TRUE(result.pause_saw_capture_stopped);
  EXPECT_TRUE(result.pause_preceded_cleanup);
  EXPECT_TRUE(result.cleanup_saw_desktop_restored);
  EXPECT_TRUE(result.cleanup_joined);
  EXPECT_TRUE(result.identity_retained);
  EXPECT_EQ(result.pause_calls, 1);
}

TEST(ArGlassesModeTransition, RecognizesTheProductionSudoVirtualDisplayHardwarePath) {
  EXPECT_TRUE(VDISPLAY::isSudoVirtualDisplayPath(
    LR"(\\?\DISPLAY#SMKD1CE#5&production&0&UID4352)"
  ));
  EXPECT_TRUE(VDISPLAY::isSudoVirtualDisplayPath(
    LR"(\\?\DISPLAY#SUDOVDA#legacy)"
  ));
  EXPECT_FALSE(VDISPLAY::isSudoVirtualDisplayPath(
    LR"(\\?\DISPLAY#TCL03D4#physical)"
  ));
}

TEST(ArGlassesModeTransition, RebasesOwnedRecoveryRectsToNewPhysicalWidth) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(
    std::move(owned),
    rect_json(7040, -1080, 8960, 0)
  ));
  const auto document = recovery_document(std::move(recoveries), 4);
  const RECT previous_original {5120, 0, 7040, 1080};
  const RECT expanded_owned_position {7040, 0, 10880, 1080};

  const auto rebased = ar_glasses::detail::rebase_topology_recovery_json_for_test(
    document.dump(),
    previous_original,
    expanded_owned_position
  );

  ASSERT_TRUE(rebased);
  const auto normalized = nlohmann::json::parse(*rebased);
  const auto &record = normalized.at("recoveries").front();
  EXPECT_EQ(record.at("original_rect"), rect_json(5120, 0, 8960, 1080));
  ASSERT_EQ(record.at("owned_rects").size(), 1u);
  EXPECT_EQ(record.at("owned_rects").front(), rect_json(7040, 0, 10880, 1080));
  EXPECT_EQ(record.at("pending_rect"), rect_json(7040, -1080, 10880, 0));
}

TEST(ArGlassesModeTransition, PreservesTheAttachedRightEdgeForALeftSideOriginal) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));
  auto document = recovery_document(std::move(recoveries), 4);
  document["recoveries"].front()["original_rect"] = rect_json(-1920, 0, 0, 1080);
  const RECT previous_original {-1920, 0, 0, 1080};
  const RECT expanded_owned_position {7040, 0, 10880, 1080};

  const auto rebased = ar_glasses::detail::rebase_topology_recovery_json_for_test(
    document.dump(),
    previous_original,
    expanded_owned_position
  );

  ASSERT_TRUE(rebased);
  const auto normalized = nlohmann::json::parse(*rebased);
  const auto &record = normalized.at("recoveries").front();
  EXPECT_EQ(record.at("original_rect"), rect_json(-3840, 0, 0, 1080));
  EXPECT_EQ(record.at("owned_rects").front(), rect_json(7040, 0, 10880, 1080));
}

TEST(ArGlassesModeTransition, RebasesOwnershipForAnUnsupportedSameOriginWidth) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));
  const auto document = recovery_document(std::move(recoveries), 4);
  const RECT previous_original {5120, 0, 7040, 1080};
  const RECT unsupported_owned_position {7040, 0, 9600, 1080};

  const auto rebased = ar_glasses::detail::rebase_topology_recovery_json_for_test(
    document.dump(),
    previous_original,
    unsupported_owned_position
  );

  ASSERT_TRUE(rebased);
  const auto normalized = nlohmann::json::parse(*rebased);
  const auto &record = normalized.at("recoveries").front();
  EXPECT_EQ(record.at("original_rect"), rect_json(5120, 0, 7680, 1080));
  EXPECT_EQ(record.at("owned_rects").front(), rect_json(7040, 0, 9600, 1080));
}

TEST(ArGlassesModeTransition, DoesNotClaimAUserMovedRectDuringModeChange) {
  auto owned = nlohmann::json::array();
  owned.emplace_back(rect_json(7040, 0, 8960, 1080));
  auto recoveries = nlohmann::json::array();
  recoveries.emplace_back(transactional_recovery_record(std::move(owned)));
  const auto document = recovery_document(std::move(recoveries), 4);
  const RECT previous_original {5120, 0, 7040, 1080};
  const RECT user_moved_expanded {8000, 200, 11840, 1280};

  EXPECT_FALSE(ar_glasses::detail::rebase_topology_recovery_json_for_test(
    document.dump(),
    previous_original,
    user_moved_expanded
  ));
}

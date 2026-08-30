#ifdef _WIN32

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
}  // namespace

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

TEST(ArGlassesPresenterPause, RequiresEveryIndependentPauseReasonToClear) {
  EXPECT_TRUE(ar_glasses::detail::local_presenter_should_run(false, false));
  EXPECT_FALSE(ar_glasses::detail::local_presenter_should_run(true, false));
  EXPECT_FALSE(ar_glasses::detail::local_presenter_should_run(false, true));
  EXPECT_FALSE(ar_glasses::detail::local_presenter_should_run(true, true));
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

  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, after));
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(after, before));
}

TEST(ArGlassesModeTransition, RebuildsOnlyForOutputAdapterOrUnsupportedChanges) {
  ar_glasses::detail::local_session_contract_t before;
  before.device_path = LR"(\\?\DISPLAY#TCL03D4#test)";
  before.adapter_id.LowPart = 42;
  before.mode = ar_glasses::presentation_mode_e::normal;
  before.hdr_known = true;

  auto changed = before;
  changed.device_path = LR"(\\?\DISPLAY#OTHER#test)";
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.adapter_id.LowPart = 43;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.hdr_active = true;
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.hdr_known = false;
  EXPECT_TRUE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.mode = ar_glasses::presentation_mode_e::unsupported;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.is_primary = true;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));

  changed = before;
  changed.is_cloned = true;
  EXPECT_FALSE(ar_glasses::detail::local_session_can_reconfigure_for_test(before, changed));
}

TEST(ArGlassesModeTransition, RetirementDoesNotFollowAReusedPhysicalTargetId) {
  ar_glasses::detail::virtual_display_identity_contract_t retiring;
  retiring.adapter_id.LowPart = 42;
  retiring.target_id = 7;
  retiring.device_path = LR"(\\?\DISPLAY#SUDOVDA#retiring)";
  retiring.gdi_name = LR"(\\.\DISPLAY5)";

  auto reused_physical = retiring;
  reused_physical.device_path = LR"(\\?\DISPLAY#TCL03D4#physical)";
  reused_physical.gdi_name = LR"(\\.\DISPLAY2)";
  reused_physical.friendly_name = L"SmartGlasses";
  EXPECT_FALSE(ar_glasses::detail::retirement_identity_matches_for_test(
    retiring,
    reused_physical
  ));

  auto renumbered_virtual = reused_physical;
  renumbered_virtual.adapter_id.LowPart = 43;
  renumbered_virtual.target_id = 9;
  renumbered_virtual.friendly_name = L"Apollo AR Desktop";
  EXPECT_FALSE(ar_glasses::detail::retirement_identity_matches_for_test(
    retiring,
    renumbered_virtual
  ));

  auto exact_sudo_identity = reused_physical;
  exact_sudo_identity.device_path = LR"(\\?\DISPLAY#SMKD1CE#replacement-path)";
  EXPECT_TRUE(ar_glasses::detail::retirement_identity_matches_for_test(
    retiring,
    exact_sudo_identity
  ));

  auto exact_apollo_identity = reused_physical;
  exact_apollo_identity.friendly_name = L"Apollo AR Desktop";
  EXPECT_TRUE(ar_glasses::detail::retirement_identity_matches_for_test(
    retiring,
    exact_apollo_identity
  ));

  auto learned_path = reused_physical;
  learned_path.device_path = retiring.device_path;
  EXPECT_TRUE(ar_glasses::detail::retirement_identity_matches_for_test(retiring, learned_path));
}

TEST(ArGlassesModeTransition, RecognizesTheProductionSudoVirtualDisplayHardwarePath) {
  EXPECT_TRUE(VDISPLAY::isSudoVirtualDisplayPathForTest(
    LR"(\\?\DISPLAY#SMKD1CE#5&production&0&UID4352)"
  ));
  EXPECT_TRUE(VDISPLAY::isSudoVirtualDisplayPathForTest(
    LR"(\\?\DISPLAY#SUDOVDA#legacy)"
  ));
  EXPECT_FALSE(VDISPLAY::isSudoVirtualDisplayPathForTest(
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

#endif

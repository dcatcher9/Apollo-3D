/**
 * @file tests/unit/platform/windows/test_ds5_sidecar_client.cpp
 * @brief Regression coverage for cancellable DS5 Core pipe shutdown.
 */
#ifdef _WIN32

  #define WIN32_LEAN_AND_MEAN
  #include "src/config.h"
  #include "src/platform/windows/ds5/ds5_controller_slot.h"
  #include "src/platform/windows/ds5/ds5_sidecar_client.h"
  #include "src/platform/windows/virtual_device_host/protocol.h"

  #include <chrono>
  #include <gtest/gtest-spi.h>
  #include <gtest/gtest.h>
  #include <windows.h>

namespace {
  using get_environment_fn_t = DWORD(WINAPI *)(LPCWSTR, LPWSTR, DWORD);
  using set_environment_fn_t = BOOL(WINAPI *)(LPCWSTR, LPCWSTR);

  struct config_scope_t {
    config_scope_t():
        settings(config::input) {}

    ~config_scope_t() {
      config::input = settings;
    }

    void enable() {
      config::input.ds5_enabled = true;
      config::input.forward_rumble = true;
    }

    config::input_t settings;
  };

  struct handle_scope_t {
    explicit handle_scope_t(HANDLE value):
        handle(value) {}

    ~handle_scope_t() {
      if (handle) {
        CloseHandle(handle);
      }
    }

    HANDLE handle;
  };

  struct event_namespace_scope_t {
    explicit event_namespace_scope_t(std::wstring_view test_name):
        suffix(std::to_wstring(GetCurrentProcessId()) + L"-" + std::to_wstring(GetTickCount64()) + L"-" + std::wstring(test_name)) {
      SetEnvironmentVariableW(L"SUNSHINE_DS5_TEST_EVENT_SUFFIX", suffix.c_str());
    }

    ~event_namespace_scope_t() {
      SetEnvironmentVariableW(L"SUNSHINE_DS5_TEST_EVENT_SUFFIX", nullptr);
    }

    std::wstring suffix;
  };

  struct environment_scope_t {
    enum class original_state_e {
      unknown,
      undefined,
      defined,
    };

    environment_scope_t(const wchar_t *variable, const wchar_t *value, get_environment_fn_t get_environment = GetEnvironmentVariableW, set_environment_fn_t set_environment = SetEnvironmentVariableW):
        variable(variable),
        set_environment(set_environment) {
      SetLastError(ERROR_SUCCESS);
      auto required = get_environment(variable, nullptr, 0);
      if (required == 0) {
        const auto error = GetLastError();
        if (error == ERROR_ENVVAR_NOT_FOUND) {
          original_state = original_state_e::undefined;
        } else if (error == ERROR_SUCCESS) {
          original_state = original_state_e::defined;
        } else {
          ADD_FAILURE() << "GetEnvironmentVariableW size query failed: " << error;
          return;
        }
      } else {
        original_state = original_state_e::defined;
      }

      while (required != 0) {
        original_value.resize(required);
        SetLastError(ERROR_SUCCESS);
        const auto copied = get_environment(variable, original_value.data(), required);
        if (copied == 0 && GetLastError() != ERROR_SUCCESS) {
          ADD_FAILURE() << "GetEnvironmentVariableW value read failed: " << GetLastError();
          original_state = original_state_e::unknown;
          original_value.clear();
          return;
        }
        if (copied < required) {
          original_value.resize(copied);
          break;
        }
        required = copied;
      }

      if (!set_environment(variable, value)) {
        ADD_FAILURE() << "SetEnvironmentVariableW scoped write failed: " << GetLastError();
        original_state = original_state_e::unknown;
        return;
      }
      restore_required = true;
    }

    ~environment_scope_t() {
      if (!restore_required) {
        return;
      }
      const auto *value = original_state == original_state_e::defined ? original_value.c_str() : nullptr;
      if (!set_environment(variable.c_str(), value)) {
        ADD_FAILURE() << "SetEnvironmentVariableW restore failed: " << GetLastError();
      }
    }

    std::wstring variable;
    std::wstring original_value;
    set_environment_fn_t set_environment;
    original_state_e original_state = original_state_e::unknown;
    bool restore_required = false;
  };

  DWORD WINAPI fail_environment_read(LPCWSTR, LPWSTR, DWORD) {
    SetLastError(ERROR_ACCESS_DENIED);
    return 0;
  }

  DWORD WINAPI fail_environment_value_read(LPCWSTR, LPWSTR value, DWORD) {
    if (!value) {
      SetLastError(ERROR_SUCCESS);
      return 2;
    }
    SetLastError(ERROR_ACCESS_DENIED);
    return 0;
  }

  BOOL WINAPI fail_environment_write(LPCWSTR, LPCWSTR) {
    SetLastError(ERROR_ACCESS_DENIED);
    return FALSE;
  }

  int environment_write_calls;

  BOOL WINAPI fail_environment_restore(LPCWSTR, LPCWSTR) {
    if (environment_write_calls++ == 0) {
      return TRUE;
    }
    SetLastError(ERROR_ACCESS_DENIED);
    return FALSE;
  }
}  // namespace

TEST(VirtualDeviceHostProtocolTests, FreezesSds5V1Abi) {
  namespace protocol = platf::virtual_device_host::protocol;
  EXPECT_EQ(protocol::MAGIC, 0x35534453u);
  EXPECT_EQ(protocol::VERSION, 1u);
  EXPECT_EQ(protocol::HEADER_SIZE, 16u);
  EXPECT_EQ(protocol::CAP_VIRTUAL_MICROPHONE, 1u << 10);
  EXPECT_EQ(protocol::CAP_PERSISTENT_DEVICE_HOST, 1u << 11);
  EXPECT_EQ(protocol::CAP_MICROPHONE_STATUS, 1u << 12);
  EXPECT_EQ(static_cast<std::uint16_t>(protocol::message_e::mic_create), 12u);
  EXPECT_EQ(static_cast<std::uint16_t>(protocol::message_e::mic_status), 107u);
  EXPECT_EQ(protocol::MIC_CREATE_PAYLOAD_SIZE, 8u);
  EXPECT_EQ(protocol::MIC_CREATE_REPLY_PAYLOAD_SIZE, 16u);
  EXPECT_EQ(protocol::MIC_OPERATION_REPLY_PAYLOAD_SIZE, 8u);
  EXPECT_EQ(protocol::MIC_PCM_HEADER_SIZE, 20u);
  EXPECT_EQ(protocol::MIC_STATUS_PAYLOAD_SIZE, 28u);
  EXPECT_EQ(protocol::MAX_MIC_PCM_FRAMES, 960u);
  EXPECT_EQ(static_cast<std::int32_t>(protocol::mic_result_e::invalid_format), -1001);
  EXPECT_EQ(static_cast<std::int32_t>(protocol::mic_result_e::device_not_created), -1004);
}

TEST(Ds5SidecarClientTests, EnvironmentScopeRestoresPreviousValues) {
  const auto variable = L"SUNSHINE_DS5_TEST_ENVIRONMENT_SCOPE_" + std::to_wstring(GetCurrentProcessId());
  EXPECT_TRUE(SetEnvironmentVariableW(variable.c_str(), nullptr));
  {
    environment_scope_t scoped(variable.c_str(), L"temporary");
  }
  SetLastError(ERROR_SUCCESS);
  EXPECT_EQ(GetEnvironmentVariableW(variable.c_str(), nullptr, 0), 0u);
  EXPECT_EQ(GetLastError(), ERROR_ENVVAR_NOT_FOUND);

  EXPECT_TRUE(SetEnvironmentVariableW(variable.c_str(), L""));
  {
    environment_scope_t scoped(variable.c_str(), L"temporary");
  }
  SetLastError(ERROR_SUCCESS);
  EXPECT_EQ(GetEnvironmentVariableW(variable.c_str(), nullptr, 0), 1u);
  EXPECT_EQ(GetLastError(), ERROR_SUCCESS);

  EXPECT_TRUE(SetEnvironmentVariableW(variable.c_str(), L"original"));
  {
    environment_scope_t scoped(variable.c_str(), L"temporary");
  }
  std::array<wchar_t, 16> restored {};
  EXPECT_EQ(GetEnvironmentVariableW(variable.c_str(), restored.data(), static_cast<DWORD>(restored.size())), 8u);
  EXPECT_STREQ(restored.data(), L"original");
  EXPECT_TRUE(SetEnvironmentVariableW(variable.c_str(), nullptr));
}

TEST(Ds5SidecarClientTests, EnvironmentScopeReportsApiFailures) {
  EXPECT_NONFATAL_FAILURE(
    { environment_scope_t scoped(L"SUNSHINE_DS5_TEST_READ_FAILURE", L"temporary", fail_environment_read, fail_environment_write); },
    "GetEnvironmentVariableW size query failed"
  );
  EXPECT_NONFATAL_FAILURE(
    { environment_scope_t scoped(L"SUNSHINE_DS5_TEST_VALUE_READ_FAILURE", L"temporary", fail_environment_value_read, fail_environment_write); },
    "GetEnvironmentVariableW value read failed"
  );
  EXPECT_NONFATAL_FAILURE(
    { environment_scope_t scoped(L"SUNSHINE_DS5_TEST_SET_FAILURE", L"temporary", GetEnvironmentVariableW, fail_environment_write); },
    "SetEnvironmentVariableW scoped write failed"
  );
  EXPECT_NONFATAL_FAILURE(
    {
      environment_write_calls = 0;
      environment_scope_t scoped(L"SUNSHINE_DS5_TEST_RESTORE_FAILURE", L"temporary", GetEnvironmentVariableW, fail_environment_restore);
    },
    "SetEnvironmentVariableW restore failed"
  );
}

TEST(Ds5SidecarClientTests, UnassignedIndexIsNotOwned) {
  platf::ds5::sidecar_client_t client;
  EXPECT_FALSE(client.owns(-1));
}

TEST(Ds5SidecarClientTests, RoutesValidatedAuthoredPcmToItsOwnBoundedQueue) {
  config_scope_t restore_config;
  restore_config.enable();
  event_namespace_scope_t events(L"authored-feedback");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, TRUE, continue_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  environment_scope_t emit_authored(L"SUNSHINE_DS5_TEST_AUTHORED_FEEDBACK", L"1");
  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-feedback", 16);
  auto haptics = mail->queue<platf::gamepad_feedback_msg_t>("ds5-pcm", 16);
  platf::ds5::sidecar_client_t client;
  ASSERT_EQ(client.alloc({2, 3}, feedback, true, false, haptics), 0);
  const auto pcm = haptics->pop(std::chrono::seconds(2));
  ASSERT_TRUE(pcm);
  EXPECT_EQ(pcm->type, platf::gamepad_feedback_e::ds5_haptics_pcm);
  EXPECT_EQ(pcm->id, 3);
  EXPECT_EQ(pcm->data.ds5_haptics.flags, 1);
  EXPECT_EQ(pcm->data.ds5_haptics.frame_count, 2);
  EXPECT_EQ(pcm->data.ds5_haptics.sequence, 0x12345678);
  EXPECT_EQ(pcm->data.ds5_haptics.presentation_time_us, 0x112345678ULL);
  for (int index = 0; index < 8; ++index) {
    EXPECT_EQ(pcm->data.ds5_haptics.pcm[index], index + 24);
  }
  auto message = feedback->pop(std::chrono::seconds(2));
  ASSERT_TRUE(message);
  EXPECT_EQ(message->type, platf::gamepad_feedback_e::set_adaptive_triggers);
  EXPECT_EQ(message->data.adaptive_triggers.left[0], 11);
  EXPECT_EQ(message->data.adaptive_triggers.right[0], 22);
  message = feedback->pop(std::chrono::seconds(2));
  ASSERT_TRUE(message);
  EXPECT_EQ(message->type, platf::gamepad_feedback_e::set_rgb_led);
  EXPECT_EQ(message->data.rgb_led.r, 10);
  message = feedback->pop(std::chrono::seconds(2));
  ASSERT_TRUE(message);
  EXPECT_EQ(message->type, platf::gamepad_feedback_e::rumble);
  client.free(2);
  EXPECT_FALSE(haptics->peek());
  EXPECT_FALSE(feedback->peek());
}

TEST(Ds5SidecarClientTests, IgnoresUnrequestedAuthoredPcmAndRetainsStandardFeedback) {
  config_scope_t restore_config;
  restore_config.enable();
  event_namespace_scope_t events(L"unrequested-authored-feedback");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, TRUE, continue_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  environment_scope_t emit_authored(L"SUNSHINE_DS5_TEST_AUTHORED_FEEDBACK", L"1");
  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-feedback", 16);
  auto haptics = mail->queue<platf::gamepad_feedback_msg_t>("ds5-pcm", 16);
  platf::ds5::sidecar_client_t client;
  ASSERT_EQ(client.alloc({2, 3}, feedback, false, false, haptics), 0);
  for (int index = 0; index < 3; ++index) {
    const auto message = feedback->pop(std::chrono::seconds(2));
    ASSERT_TRUE(message);
    EXPECT_NE(message->type, platf::gamepad_feedback_e::ds5_haptics_pcm);
  }
  client.free(2);
  EXPECT_FALSE(haptics->peek());
}

TEST(Ds5SidecarClientTests, AllocThenFreeCancelsBlockedReader) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"blocked-reader");
  const auto reader_name = L"Local\\sunshine-ds5-test-reader-" + events.suffix;
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto marker_name = L"Local\\sunshine-ds5-test-marker-" + events.suffix;
  handle_scope_t reader_event(CreateEventW(nullptr, FALSE, FALSE, reader_name.c_str()));
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t marker_event(CreateEventW(nullptr, FALSE, FALSE, marker_name.c_str()));
  ASSERT_NE(reader_event.handle, nullptr);
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(marker_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-lifecycle-test");
  auto feedback_for_test = feedback;
  platf::ds5::sidecar_client_t client;
  ASSERT_EQ(client.alloc({0, 0}, std::move(feedback), false), 0);

  ASSERT_EQ(WaitForSingleObject(reader_event.handle, 2000), WAIT_OBJECT_0);
  ASSERT_TRUE(SetEvent(continue_event.handle));
  ASSERT_EQ(WaitForSingleObject(marker_event.handle, 2000), WAIT_OBJECT_0);
  const auto marker = feedback_for_test->pop(std::chrono::seconds(2));
  ASSERT_TRUE(marker);
  ASSERT_EQ(marker->type, platf::gamepad_feedback_e::rumble);
  // The second signal is raised only after the marker was processed and the
  // following overlapped read is actually pending.
  ASSERT_EQ(WaitForSingleObject(reader_event.handle, 2000), WAIT_OBJECT_0);

  const auto started = std::chrono::steady_clock::now();
  client.free(0);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_LT(elapsed, std::chrono::seconds(2));
}

TEST(Ds5SidecarClientTests, AttachSurvivesInterleavedAsyncFeedback) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"interleaved-feedback");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto marker_name = L"Local\\sunshine-ds5-test-marker-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t marker_event(CreateEventW(nullptr, FALSE, FALSE, marker_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(marker_event.handle, nullptr);
  ASSERT_NE(SetEnvironmentVariableW(L"SUNSHINE_DS5_TEST_INTERLEAVE", L"1"), 0);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-interleave-test");
  auto feedback_for_test = feedback;
  platf::ds5::sidecar_client_t client;
  // The fake peer emits an async rumble ahead of the attach reply. The
  // transaction must dispatch it and still match the reply; before the
  // multiplexing fix the rumble was misread as the reply and alloc failed.
  EXPECT_EQ(client.alloc({0, 0}, std::move(feedback), false), 0);
  SetEnvironmentVariableW(L"SUNSHINE_DS5_TEST_INTERLEAVE", nullptr);
  const auto early = feedback_for_test->pop(std::chrono::seconds(2));
  ASSERT_TRUE(early);
  EXPECT_EQ(early->type, platf::gamepad_feedback_e::rumble);

  ASSERT_TRUE(SetEvent(continue_event.handle));
  ASSERT_EQ(WaitForSingleObject(marker_event.handle, 2000), WAIT_OBJECT_0);
  const auto marker = feedback_for_test->pop(std::chrono::seconds(2));
  ASSERT_TRUE(marker);
  EXPECT_EQ(marker->type, platf::gamepad_feedback_e::rumble);
  client.free(0);
}

TEST(Ds5SidecarClientTests, RejectsCompositeAttachWithoutAudioEndpoint) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"audio-endpoint");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, TRUE, continue_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-audio-attach-test");
  platf::ds5::sidecar_client_t client;
  EXPECT_EQ(client.alloc({0, 0}, std::move(feedback), true), -1);
  EXPECT_FALSE(client.owns(0));
}

TEST(Ds5SidecarClientTests, FallsBackToHidWhenPeerLacksAudioPolicyCapability) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"legacy-capabilities");
  environment_scope_t legacy_capabilities(L"SUNSHINE_DS5_TEST_LEGACY_CAPABILITIES", L"1");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, TRUE, continue_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-legacy-capabilities-test");
  platf::ds5::sidecar_client_t client;
  EXPECT_EQ(client.alloc({0, 0}, std::move(feedback), true), 0);
  EXPECT_TRUE(client.owns(0));
  client.free(0);
}

TEST(Ds5SidecarClientTests, SendsNegotiatedGenshinCompatibilityAttachFlag) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"genshin-compatibility");
  environment_scope_t enable_compatibility(
    L"SUNSHINE_DS5_TEST_GENSHIN_COMPATIBILITY",
    L"1"
  );
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto compatibility_name =
    L"Local\\sunshine-ds5-test-genshin-compatibility-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, TRUE, continue_name.c_str()));
  handle_scope_t compatibility_event(CreateEventW(nullptr, FALSE, FALSE, compatibility_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(compatibility_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-genshin-compatibility-test");
  platf::ds5::sidecar_client_t client;
  ASSERT_EQ(client.alloc({0, 0}, std::move(feedback), true, true), 0);
  EXPECT_EQ(WaitForSingleObject(compatibility_event.handle, 2000), WAIT_OBJECT_0);
  client.free(0);
}

TEST(Ds5SidecarClientTests, RejectsGenshinCompatibilityWithoutSidecarCapability) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"genshin-capability-required");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, TRUE, continue_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-genshin-capability-test");
  platf::ds5::sidecar_client_t client;
  EXPECT_EQ(client.alloc({0, 0}, std::move(feedback), true, true), -1);
  EXPECT_FALSE(client.owns(0));
}

TEST(Ds5SidecarClientTests, RelaunchesOnceAfterUnexpectedExit) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"recover-once");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto crash_name = L"Local\\sunshine-ds5-test-crash-once-" + events.suffix;
  const auto recovered_name = L"Local\\sunshine-ds5-test-recovered-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t crash_event(CreateEventW(nullptr, TRUE, FALSE, crash_name.c_str()));
  handle_scope_t recovered_event(CreateEventW(nullptr, FALSE, FALSE, recovered_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(crash_event.handle, nullptr);
  ASSERT_NE(recovered_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-recovery-test");
  auto feedback_for_test = feedback;
  platf::ds5::sidecar_client_t client;
  ASSERT_EQ(client.alloc({0, 0}, std::move(feedback), false), 0);
  ASSERT_EQ(WaitForSingleObject(recovered_event.handle, 5000), WAIT_OBJECT_0);
  ASSERT_TRUE(SetEvent(continue_event.handle));
  const auto marker = feedback_for_test->pop(std::chrono::seconds(2));
  ASSERT_TRUE(marker);
  ASSERT_EQ(marker->type, platf::gamepad_feedback_e::rumble);
  EXPECT_TRUE(client.owns(0));
  client.free(0);
}

TEST(Ds5SidecarClientTests, FallsBackToHidOnlyWhenVirtualAudioBecomesDefault) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"audio-policy-fallback");
  environment_scope_t enable_policy_fallback(L"SUNSHINE_DS5_TEST_AUDIO_POLICY_FALLBACK", L"1");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto policy_once_name = L"Local\\sunshine-ds5-test-policy-once-" + events.suffix;
  const auto hid_fallback_name = L"Local\\sunshine-ds5-test-hid-fallback-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t policy_once_event(CreateEventW(nullptr, TRUE, FALSE, policy_once_name.c_str()));
  handle_scope_t hid_fallback_event(CreateEventW(nullptr, FALSE, FALSE, hid_fallback_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(policy_once_event.handle, nullptr);
  ASSERT_NE(hid_fallback_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-audio-policy-fallback-test");
  auto feedback_for_test = feedback;
  platf::ds5::sidecar_client_t client;
  ASSERT_EQ(client.alloc({0, 0}, std::move(feedback), true), 0);
  ASSERT_EQ(WaitForSingleObject(hid_fallback_event.handle, 5000), WAIT_OBJECT_0);
  ASSERT_TRUE(SetEvent(continue_event.handle));
  const auto marker = feedback_for_test->pop(std::chrono::seconds(2));
  ASSERT_TRUE(marker);
  EXPECT_EQ(marker->type, platf::gamepad_feedback_e::rumble);
  EXPECT_TRUE(client.owns(0));
  client.free(0);
}

TEST(Ds5SidecarClientTests, FreeCancelsPendingRecovery) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"cancel-recovery");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto crash_name = L"Local\\sunshine-ds5-test-crash-once-" + events.suffix;
  const auto recovery_started_name = L"Local\\sunshine-ds5-test-recovery-started-" + events.suffix;
  const auto recovery_wait_name = L"Local\\sunshine-ds5-test-recovery-wait-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t crash_event(CreateEventW(nullptr, TRUE, FALSE, crash_name.c_str()));
  handle_scope_t recovery_started_event(CreateEventW(nullptr, FALSE, FALSE, recovery_started_name.c_str()));
  handle_scope_t recovery_wait_event(CreateEventW(nullptr, TRUE, FALSE, recovery_wait_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(crash_event.handle, nullptr);
  ASSERT_NE(recovery_started_event.handle, nullptr);
  ASSERT_NE(recovery_wait_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-cancel-recovery-test");
  platf::ds5::sidecar_client_t client;
  ASSERT_EQ(client.alloc({0, 0}, std::move(feedback), false), 0);
  ASSERT_EQ(WaitForSingleObject(recovery_started_event.handle, 5000), WAIT_OBJECT_0);

  // The transport is offline, but the sidecar still owns the controller id.
  // The input layer relies on owns() to route release to sidecar_client_t::free().
  EXPECT_TRUE(client.owns(0));
  const auto started = std::chrono::steady_clock::now();
  client.free(0);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_LT(elapsed, std::chrono::seconds(3));
  EXPECT_FALSE(client.owns(0));
}

TEST(Ds5SidecarClientTests, ReallocatesAfterRecoveryFailure) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"recovery-failure");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto crash_name = L"Local\\sunshine-ds5-test-crash-always-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t crash_event(CreateEventW(nullptr, FALSE, FALSE, crash_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(crash_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("ds5-recovery-failure-test");
  platf::ds5::sidecar_client_t client;
  ASSERT_EQ(client.alloc({0, 0}, std::move(feedback), false), 0);

  // Both the initial sidecar and its one recovery attempt exit immediately.
  ASSERT_EQ(WaitForSingleObject(crash_event.handle, 5000), WAIT_OBJECT_0);
  ASSERT_EQ(WaitForSingleObject(crash_event.handle, 5000), WAIT_OBJECT_0);

  int result = -1;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  for (unsigned attempt = 0; result < 0 && std::chrono::steady_clock::now() < deadline; ++attempt) {
    auto retry_feedback = mail->queue<platf::gamepad_feedback_msg_t>(
      "ds5-recovery-failure-retry-" + std::to_string(attempt)
    );
    result = client.alloc({1, 0}, std::move(retry_feedback), false);
    if (result < 0) {
      Sleep(10);
    }
  }
  EXPECT_EQ(result, 0);
}

TEST(Ds5ControllerSlotTests, FailedRecoveryFallsBackOnceWithOriginalControllerAndQueues) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"slot-fallback");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto crash_name = L"Local\\sunshine-ds5-test-crash-always-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t crash_event(CreateEventW(nullptr, FALSE, FALSE, crash_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(crash_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("slot-feedback");
  auto haptics = mail->queue<platf::gamepad_feedback_msg_t>("slot-haptics");
  platf::gamepad_id_t id {2, 5};
  platf::gamepad_arrival_t metadata {2, 0x220, 0x12345678, haptics};
  platf::ds5::controller_slot_t slot;
  ASSERT_EQ(slot.alloc(id, metadata, feedback, false), 0);
  // The slot must retain values, including the session queues, rather than
  // borrowing the caller's arrival packet or using its global index as client id.
  id = {7, 8};
  metadata = {};

  ASSERT_EQ(WaitForSingleObject(crash_event.handle, 5000), WAIT_OBJECT_0);
  ASSERT_EQ(WaitForSingleObject(crash_event.handle, 5000), WAIT_OBJECT_0);
  int fallback_calls = 0;
  auto fallback = [&](const platf::gamepad_id_t &saved_id, const platf::gamepad_arrival_t &saved_metadata, platf::feedback_queue_t saved_feedback) {
    ++fallback_calls;
    EXPECT_EQ(saved_id.globalIndex, 2);
    EXPECT_EQ(saved_id.clientRelativeIndex, 5);
    EXPECT_EQ(saved_metadata.type, 2);
    EXPECT_EQ(saved_metadata.capabilities, 0x220);
    EXPECT_EQ(saved_metadata.supportedButtons, 0x12345678u);
    EXPECT_EQ(saved_metadata.haptics_feedback_queue, haptics);
    EXPECT_EQ(saved_feedback, feedback);
    return 0;
  };
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (fallback_calls == 0 && std::chrono::steady_clock::now() < deadline) {
    slot.sidecar_for_input(fallback);
    if (fallback_calls == 0) {
      Sleep(10);
    }
  }
  ASSERT_EQ(fallback_calls, 1);
  // All later state/touch/motion/battery routes must use the existing fallback
  // controller, without reattaching a helper or allocating another ViGEm target.
  for (int event = 0; event < 4; ++event) {
    EXPECT_EQ(slot.sidecar_for_input(fallback), nullptr);
  }
  EXPECT_EQ(fallback_calls, 1);
  slot.reset();
  EXPECT_EQ(slot.sidecar_for_input(fallback), nullptr);
  EXPECT_EQ(fallback_calls, 1);
}

TEST(Ds5ControllerSlotTests, FailedFallbackRetainsSessionAndThrottlesRetry) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"slot-retry");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto crash_name = L"Local\\sunshine-ds5-test-crash-always-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t crash_event(CreateEventW(nullptr, FALSE, FALSE, crash_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(crash_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback = mail->queue<platf::gamepad_feedback_msg_t>("slot-retry-feedback");
  platf::ds5::controller_slot_t slot;
  ASSERT_EQ(slot.alloc({3, 6}, {}, feedback, false), 0);
  ASSERT_EQ(WaitForSingleObject(crash_event.handle, 5000), WAIT_OBJECT_0);
  ASSERT_EQ(WaitForSingleObject(crash_event.handle, 5000), WAIT_OBJECT_0);

  int fallback_calls = 0;
  auto fallback = [&](const platf::gamepad_id_t &id, const platf::gamepad_arrival_t &, platf::feedback_queue_t saved_feedback) {
    EXPECT_EQ(id.globalIndex, 3);
    EXPECT_EQ(id.clientRelativeIndex, 6);
    EXPECT_EQ(saved_feedback, feedback);
    return ++fallback_calls == 1 ? -1 : 0;
  };
  const auto now = platf::ds5::controller_slot_t::clock_t::now();
  const auto deadline = now + std::chrono::seconds(5);
  while (fallback_calls == 0 && std::chrono::steady_clock::now() < deadline) {
    slot.sidecar_for_input(fallback, now);
    if (fallback_calls == 0) {
      Sleep(10);
    }
  }
  ASSERT_EQ(fallback_calls, 1);
  EXPECT_EQ(slot.sidecar_for_input(fallback, now), nullptr);
  EXPECT_EQ(slot.sidecar_for_input(fallback, now + std::chrono::milliseconds(999)), nullptr);
  EXPECT_EQ(fallback_calls, 1);
  EXPECT_EQ(slot.sidecar_for_input(fallback, now + std::chrono::seconds(1)), nullptr);
  EXPECT_EQ(fallback_calls, 2);
  EXPECT_EQ(slot.sidecar_for_input(fallback, now + std::chrono::seconds(2)), nullptr);
  EXPECT_EQ(fallback_calls, 2);
}

TEST(Ds5ControllerSlotTests, ResetDuringRecoveryPreventsFallbackIntoReplacementSession) {
  config_scope_t restore_config;
  restore_config.enable();

  event_namespace_scope_t events(L"slot-cancel-recovery");
  const auto continue_name = L"Local\\sunshine-ds5-test-continue-" + events.suffix;
  const auto crash_name = L"Local\\sunshine-ds5-test-crash-once-" + events.suffix;
  const auto recovery_started_name = L"Local\\sunshine-ds5-test-recovery-started-" + events.suffix;
  const auto recovery_wait_name = L"Local\\sunshine-ds5-test-recovery-wait-" + events.suffix;
  handle_scope_t continue_event(CreateEventW(nullptr, FALSE, FALSE, continue_name.c_str()));
  handle_scope_t crash_event(CreateEventW(nullptr, TRUE, FALSE, crash_name.c_str()));
  handle_scope_t recovery_started_event(CreateEventW(nullptr, FALSE, FALSE, recovery_started_name.c_str()));
  handle_scope_t recovery_wait_event(CreateEventW(nullptr, TRUE, FALSE, recovery_wait_name.c_str()));
  ASSERT_NE(continue_event.handle, nullptr);
  ASSERT_NE(crash_event.handle, nullptr);
  ASSERT_NE(recovery_started_event.handle, nullptr);
  ASSERT_NE(recovery_wait_event.handle, nullptr);

  auto mail = std::make_shared<safe::mail_raw_t>();
  auto feedback_a = mail->queue<platf::gamepad_feedback_msg_t>("slot-session-a");
  auto feedback_b = mail->queue<platf::gamepad_feedback_msg_t>("slot-session-b");
  platf::ds5::controller_slot_t slot;
  ASSERT_EQ(slot.alloc({1, 6}, {}, feedback_a, false), 0);
  ASSERT_EQ(WaitForSingleObject(recovery_started_event.handle, 5000), WAIT_OBJECT_0);

  int fallback_calls = 0;
  auto fallback = [&](const platf::gamepad_id_t &, const platf::gamepad_arrival_t &, platf::feedback_queue_t) {
    ++fallback_calls;
    return 0;
  };
  auto *recovering = slot.sidecar_for_input(fallback);
  ASSERT_NE(recovering, nullptr);
  EXPECT_TRUE(recovering->owns(1));
  EXPECT_EQ(fallback_calls, 0);
  const auto started = std::chrono::steady_clock::now();
  slot.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - started, std::chrono::seconds(3));
  EXPECT_EQ(slot.sidecar_for_input(fallback), nullptr);
  EXPECT_EQ(fallback_calls, 0);

  ASSERT_TRUE(SetEvent(recovery_wait_event.handle));
  ASSERT_TRUE(SetEvent(continue_event.handle));
  ASSERT_EQ(slot.alloc({4, 2}, {}, feedback_b, false), 0);
  auto *replacement = slot.sidecar_for_input(fallback);
  ASSERT_NE(replacement, nullptr);
  EXPECT_TRUE(replacement->owns(4));
  EXPECT_FALSE(replacement->owns(1));
  EXPECT_EQ(fallback_calls, 0);
  slot.reset();
}

#endif

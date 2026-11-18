/**
 * @file tests/unit/platform/windows/test_nvapi_wrapper.cpp
 * @brief Exercise the production NVAPI wrapper without loading the NVIDIA driver.
 */

#include "src/platform/windows/nvprefs/driver_settings.h"
#include "src/platform/windows/nvprefs/nvapi_test_hooks.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
  namespace sdk_reference {
// The SDK header defines storage, so keep this independent reference out of global scope.
#include <nvapi_interface.h>
  }  // namespace sdk_reference

  struct interface_t {
    const char *name;
    NvU32 id;
  };

  // Independent contracts from the vendored NVAPI SDK's interface table.
  constexpr std::array expected_interfaces {
    interface_t {"NvAPI_GetErrorMessage", 0x6c2d048c},
    interface_t {"NvAPI_DRS_CreateSession", 0x0694d52e},
    interface_t {"NvAPI_DRS_DestroySession", 0xdad9cff8},
    interface_t {"NvAPI_DRS_LoadSettings", 0x375dbd6b},
    interface_t {"NvAPI_DRS_SaveSettings", 0xfcbc7e14},
    interface_t {"NvAPI_DRS_CreateProfile", 0xcc176068},
    interface_t {"NvAPI_DRS_FindProfileByName", 0x7e4a9a0b},
    interface_t {"NvAPI_DRS_CreateApplication", 0x4347a9de},
    interface_t {"NvAPI_DRS_GetApplicationInfo", 0xed1f8c69},
    interface_t {"NvAPI_DRS_SetSetting", 0x577dd202},
    interface_t {"NvAPI_DRS_GetSetting", 0x73bf8338},
    interface_t {"NvAPI_DRS_DeleteProfileSetting", 0xe4a26362},
    interface_t {"NvAPI_DRS_GetBaseProfile", 0xda8466a0},
  };
  // Resolve exactly the interfaces exposed by this wrapper, with no unrelated SDK queries.
  constexpr std::size_t expected_resolver_calls = 13;

  struct forwarded_call_t {
    NvU32 id {};
    NvDRSSessionHandle session {};
    NvDRSProfileHandle profile {};
    const void *first {};
    const void *second {};
    NvU32 value {};
  };

  struct loader_state_t {
    bool library_available = true;
    bool query_available = true;
    std::optional<NvU32> missing_function;
    unsigned loads = 0;
    unsigned proc_lookups = 0;
    unsigned frees = 0;
    unsigned native_initializes = 0;
    unsigned native_unloads = 0;
    std::string loaded_name;
    HANDLE load_file {};
    DWORD load_flags {};
    std::vector<NvU32> resolved_ids;
    std::vector<forwarded_call_t> calls;
    NvAPI_Status forwarded_status = NVAPI_ERROR;
    NvDRSSessionHandle__ session_storage {};
    NvDRSProfileHandle__ profile_storage {};
    char module_storage {};

    HMODULE module() {
      return reinterpret_cast<HMODULE>(&module_storage);
    }

    NvAPI_Status record(std::size_t index, NvDRSSessionHandle session = nullptr, NvDRSProfileHandle profile = nullptr, const void *first = nullptr, const void *second = nullptr, NvU32 value = 0) {
      calls.push_back({expected_interfaces[index].id, session, profile, first, second, value});
      return forwarded_status;
    }
  };

  loader_state_t *active_state = nullptr;

  NvAPI_Status __cdecl fake_native_initialize() {
    ++active_state->native_initializes;
    return NVAPI_ERROR;
  }

  NvAPI_Status __cdecl fake_native_unload() {
    ++active_state->native_unloads;
    return NVAPI_ERROR;
  }

  NvAPI_Status __cdecl fake_get_error_message(NvAPI_Status nr, NvAPI_ShortString description) {
    constexpr char result[] = "fake NVAPI error";
    std::memcpy(description, result, sizeof(result));
    return active_state->record(0, nullptr, nullptr, description, nullptr, static_cast<NvU32>(nr));
  }

  NvAPI_Status __cdecl fake_create_session(NvDRSSessionHandle *session) {
    *session = &active_state->session_storage;
    return active_state->record(1, nullptr, nullptr, session);
  }

  NvAPI_Status __cdecl fake_destroy_session(NvDRSSessionHandle session) {
    return active_state->record(2, session);
  }

  NvAPI_Status __cdecl fake_load_settings(NvDRSSessionHandle session) {
    return active_state->record(3, session);
  }

  NvAPI_Status __cdecl fake_save_settings(NvDRSSessionHandle session) {
    return active_state->record(4, session);
  }

  NvAPI_Status __cdecl fake_create_profile(NvDRSSessionHandle session, NVDRS_PROFILE *info, NvDRSProfileHandle *profile) {
    *profile = &active_state->profile_storage;
    return active_state->record(5, session, nullptr, info, profile);
  }

  NvAPI_Status __cdecl fake_find_profile(NvDRSSessionHandle session, NvAPI_UnicodeString name, NvDRSProfileHandle *profile) {
    *profile = &active_state->profile_storage;
    return active_state->record(6, session, nullptr, name, profile);
  }

  NvAPI_Status __cdecl fake_create_application(NvDRSSessionHandle session, NvDRSProfileHandle profile, NVDRS_APPLICATION *application) {
    return active_state->record(7, session, profile, application);
  }

  NvAPI_Status __cdecl fake_get_application(NvDRSSessionHandle session, NvDRSProfileHandle profile, NvAPI_UnicodeString name, NVDRS_APPLICATION *application) {
    application->isPredefined = 1;
    return active_state->record(8, session, profile, name, application);
  }

  NvAPI_Status __cdecl fake_set_setting(NvDRSSessionHandle session, NvDRSProfileHandle profile, NVDRS_SETTING *setting) {
    return active_state->record(9, session, profile, setting);
  }

  NvAPI_Status __cdecl fake_get_setting(NvDRSSessionHandle session, NvDRSProfileHandle profile, NvU32 id, NVDRS_SETTING *setting) {
    setting->u32CurrentValue = 0xfedcba98;
    return active_state->record(10, session, profile, setting, nullptr, id);
  }

  NvAPI_Status __cdecl fake_delete_setting(NvDRSSessionHandle session, NvDRSProfileHandle profile, NvU32 id) {
    return active_state->record(11, session, profile, nullptr, nullptr, id);
  }

  NvAPI_Status __cdecl fake_get_base_profile(NvDRSSessionHandle session, NvDRSProfileHandle *profile) {
    *profile = &active_state->profile_storage;
    return active_state->record(12, session, nullptr, profile);
  }

  static_assert(std::is_same_v<decltype(&fake_get_error_message), decltype(&NvAPI_GetErrorMessage)>);
  static_assert(std::is_same_v<decltype(&fake_create_session), decltype(&NvAPI_DRS_CreateSession)>);
  static_assert(std::is_same_v<decltype(&fake_destroy_session), decltype(&NvAPI_DRS_DestroySession)>);
  static_assert(std::is_same_v<decltype(&fake_load_settings), decltype(&NvAPI_DRS_LoadSettings)>);
  static_assert(std::is_same_v<decltype(&fake_save_settings), decltype(&NvAPI_DRS_SaveSettings)>);
  static_assert(std::is_same_v<decltype(&fake_create_profile), decltype(&NvAPI_DRS_CreateProfile)>);
  static_assert(std::is_same_v<decltype(&fake_find_profile), decltype(&NvAPI_DRS_FindProfileByName)>);
  static_assert(std::is_same_v<decltype(&fake_create_application), decltype(&NvAPI_DRS_CreateApplication)>);
  static_assert(std::is_same_v<decltype(&fake_get_application), decltype(&NvAPI_DRS_GetApplicationInfo)>);
  static_assert(std::is_same_v<decltype(&fake_set_setting), decltype(&NvAPI_DRS_SetSetting)>);
  static_assert(std::is_same_v<decltype(&fake_get_setting), decltype(&NvAPI_DRS_GetSetting)>);
  static_assert(std::is_same_v<decltype(&fake_delete_setting), decltype(&NvAPI_DRS_DeleteProfileSetting)>);
  static_assert(std::is_same_v<decltype(&fake_get_base_profile), decltype(&NvAPI_DRS_GetBaseProfile)>);
  static_assert(std::is_same_v<decltype(&fake_native_initialize), decltype(&NvAPI_Initialize)>);
  static_assert(std::is_same_v<decltype(&fake_native_unload), decltype(&NvAPI_Unload)>);

  void *__cdecl fake_query_interface(NvU32 id) {
    active_state->resolved_ids.push_back(id);
    if (active_state->missing_function == id) {
      return nullptr;
    }
    switch (id) {
      case 0x0150e828:
        return reinterpret_cast<void *>(&fake_native_initialize);
      case 0xd22bdd7e:
        return reinterpret_cast<void *>(&fake_native_unload);
      case expected_interfaces[0].id:
        return reinterpret_cast<void *>(&fake_get_error_message);
      case expected_interfaces[1].id:
        return reinterpret_cast<void *>(&fake_create_session);
      case expected_interfaces[2].id:
        return reinterpret_cast<void *>(&fake_destroy_session);
      case expected_interfaces[3].id:
        return reinterpret_cast<void *>(&fake_load_settings);
      case expected_interfaces[4].id:
        return reinterpret_cast<void *>(&fake_save_settings);
      case expected_interfaces[5].id:
        return reinterpret_cast<void *>(&fake_create_profile);
      case expected_interfaces[6].id:
        return reinterpret_cast<void *>(&fake_find_profile);
      case expected_interfaces[7].id:
        return reinterpret_cast<void *>(&fake_create_application);
      case expected_interfaces[8].id:
        return reinterpret_cast<void *>(&fake_get_application);
      case expected_interfaces[9].id:
        return reinterpret_cast<void *>(&fake_set_setting);
      case expected_interfaces[10].id:
        return reinterpret_cast<void *>(&fake_get_setting);
      case expected_interfaces[11].id:
        return reinterpret_cast<void *>(&fake_delete_setting);
      case expected_interfaces[12].id:
        return reinterpret_cast<void *>(&fake_get_base_profile);
      default:
        return nullptr;
    }
  }

  HMODULE WINAPI fake_load_library(LPCSTR name, HANDLE file, DWORD flags) {
    ++active_state->loads;
    active_state->loaded_name = name;
    active_state->load_file = file;
    active_state->load_flags = flags;
    return active_state->library_available ? active_state->module() : nullptr;
  }

  FARPROC WINAPI fake_get_proc_address(HMODULE module, LPCSTR name) {
    ++active_state->proc_lookups;
    EXPECT_EQ(module, active_state->module());
    EXPECT_EQ(std::string_view(name), "nvapi_QueryInterface");
    return active_state->query_available ? reinterpret_cast<FARPROC>(&fake_query_interface) : nullptr;
  }

  BOOL WINAPI fake_free_library(HMODULE module) {
    ++active_state->frees;
    EXPECT_EQ(module, active_state->module());
    return TRUE;
  }

  constexpr nvprefs::detail::nvapi_loader_api_t fake_loader {
    &fake_load_library,
    &fake_get_proc_address,
    &fake_free_library,
  };

  class NvapiWrapperTest: public testing::Test {
  protected:
    void SetUp() override {
      active_state = &state;
      ASSERT_TRUE(nvprefs::detail::set_nvapi_loader_for_test(fake_loader));
    }

    void TearDown() override {
      // Leave the fake loader installed until the real wrapper releases its fake module.
      EXPECT_EQ(NvAPI_Unload(), NVAPI_OK);
      EXPECT_EQ(state.native_initializes, 0u);
      EXPECT_EQ(state.native_unloads, 0u);
      RecordProperty("native_initialize_calls", std::to_string(state.native_initializes));
      RecordProperty("native_unload_calls", std::to_string(state.native_unloads));
      EXPECT_TRUE(nvprefs::detail::reset_nvapi_loader_for_test());
      active_state = nullptr;
    }

    NvAPI_Status invoke(std::size_t index) {
      const auto session = &state.session_storage;
      const auto profile = &state.profile_storage;
      switch (index) {
        case 0:
          return NvAPI_GetErrorMessage(NVAPI_INVALID_ARGUMENT, description);
        case 1:
          return NvAPI_DRS_CreateSession(&output_session);
        case 2:
          return NvAPI_DRS_DestroySession(session);
        case 3:
          return NvAPI_DRS_LoadSettings(session);
        case 4:
          return NvAPI_DRS_SaveSettings(session);
        case 5:
          return NvAPI_DRS_CreateProfile(session, &profile_info, &output_profile);
        case 6:
          return NvAPI_DRS_FindProfileByName(session, unicode_name, &output_profile);
        case 7:
          return NvAPI_DRS_CreateApplication(session, profile, &application);
        case 8:
          return NvAPI_DRS_GetApplicationInfo(session, profile, unicode_name, &application);
        case 9:
          return NvAPI_DRS_SetSetting(session, profile, &setting);
        case 10:
          return NvAPI_DRS_GetSetting(session, profile, setting_id, &setting);
        case 11:
          return NvAPI_DRS_DeleteProfileSetting(session, profile, setting_id);
        case 12:
          return NvAPI_DRS_GetBaseProfile(session, &output_profile);
        default:
          ADD_FAILURE() << "Unknown wrapped interface";
          return NVAPI_ERROR;
      }
    }

    loader_state_t state;
    NvAPI_ShortString description {};
    NvAPI_UnicodeString unicode_name {u'P', u'r', u'\u00f8', u'f', u'i', u'l', u'e', 0};
    NVDRS_PROFILE profile_info {};
    NVDRS_APPLICATION application {};
    NVDRS_SETTING setting {};
    NvDRSSessionHandle output_session {};
    NvDRSProfileHandle output_profile {};
    static constexpr NvU32 setting_id = 0xf1234567;
  };

  TEST_F(NvapiWrapperTest, LoadsFromSystem32AndResolvesEveryWrappedSdkIdExactlyOnce) {
    ASSERT_EQ(NvAPI_Initialize(), NVAPI_OK);
    EXPECT_EQ(state.loads, 1u);
    EXPECT_EQ(state.proc_lookups, 1u);
    EXPECT_EQ(state.load_file, nullptr);
    EXPECT_EQ(state.load_flags, static_cast<DWORD>(LOAD_LIBRARY_SEARCH_SYSTEM32));
#ifdef _WIN64
    EXPECT_EQ(state.loaded_name, "nvapi64.dll");
#else
    EXPECT_EQ(state.loaded_name, "nvapi.dll");
#endif
    RecordProperty("query_interface_calls", std::to_string(state.resolved_ids.size()));
    RecordProperty("wrapped_interface_count", std::to_string(expected_interfaces.size()));
    EXPECT_EQ(state.resolved_ids.size(), expected_resolver_calls);
    for (const auto &entry : expected_interfaces) {
      SCOPED_TRACE(entry.name);
      const auto matches_name = [&](const auto &sdk) {
        return std::string_view(sdk.func) == entry.name;
      };
      ASSERT_EQ(std::count_if(std::begin(sdk_reference::nvapi_interface_table), std::end(sdk_reference::nvapi_interface_table), matches_name), 1);
      const auto sdk = std::find_if(std::begin(sdk_reference::nvapi_interface_table), std::end(sdk_reference::nvapi_interface_table), matches_name);
      EXPECT_EQ(sdk->id, entry.id);
      EXPECT_EQ(std::count(state.resolved_ids.begin(), state.resolved_ids.end(), entry.id), 1);
    }
    EXPECT_TRUE(state.calls.empty());
  }

  TEST_F(NvapiWrapperTest, EveryTypedForwarderPreservesArgumentsOutputAndStatus) {
    ASSERT_EQ(NvAPI_Initialize(), NVAPI_OK);
    profile_info.version = NVDRS_PROFILE_VER;
    application.version = NVDRS_APPLICATION_VER;
    setting.version = NVDRS_SETTING_VER;
    for (const auto status : {NVAPI_OK, NVAPI_ERROR}) {
      state.forwarded_status = status;
      for (std::size_t index = 0; index < expected_interfaces.size(); ++index) {
        SCOPED_TRACE(expected_interfaces[index].name);
        const auto calls = state.calls.size();
        EXPECT_EQ(invoke(index), status);
        ASSERT_EQ(state.calls.size(), calls + 1);
        const auto &call = state.calls.back();
        EXPECT_EQ(call.id, expected_interfaces[index].id);
        if (index >= 2) {
          EXPECT_EQ(call.session, &state.session_storage);
        }
        if (index >= 7 && index <= 11) {
          EXPECT_EQ(call.profile, &state.profile_storage);
        }
        switch (index) {
          case 0:
            EXPECT_EQ(call.first, description);
            EXPECT_EQ(call.value, static_cast<NvU32>(NVAPI_INVALID_ARGUMENT));
            EXPECT_STREQ(description, "fake NVAPI error");
            break;
          case 1:
            EXPECT_EQ(call.first, &output_session);
            EXPECT_EQ(output_session, &state.session_storage);
            break;
          case 5:
            EXPECT_EQ(call.first, &profile_info);
            EXPECT_EQ(call.second, &output_profile);
            EXPECT_EQ(output_profile, &state.profile_storage);
            EXPECT_EQ(profile_info.version, NVDRS_PROFILE_VER);
            break;
          case 6:
            EXPECT_EQ(call.first, unicode_name);
            EXPECT_EQ(call.second, &output_profile);
            EXPECT_EQ(output_profile, &state.profile_storage);
            break;
          case 7:
            EXPECT_EQ(call.first, &application);
            EXPECT_EQ(application.version, NVDRS_APPLICATION_VER);
            break;
          case 8:
            EXPECT_EQ(call.first, unicode_name);
            EXPECT_EQ(call.second, &application);
            EXPECT_EQ(application.isPredefined, 1u);
            break;
          case 9:
            EXPECT_EQ(call.first, &setting);
            EXPECT_EQ(setting.version, NVDRS_SETTING_VER);
            break;
          case 10:
            EXPECT_EQ(call.first, &setting);
            EXPECT_EQ(call.value, setting_id);
            EXPECT_EQ(setting.u32CurrentValue, 0xfedcba98u);
            break;
          case 11:
            EXPECT_EQ(call.value, setting_id);
            break;
          case 12:
            EXPECT_EQ(call.first, &output_profile);
            EXPECT_EQ(output_profile, &state.profile_storage);
            break;
          default:
            break;
        }
      }
    }
  }

  TEST_F(NvapiWrapperTest, UninitializedForwardersReturnTheExistingNotSupportedStatusWithoutLoading) {
    for (std::size_t index = 0; index < expected_interfaces.size(); ++index) {
      SCOPED_TRACE(expected_interfaces[index].name);
      EXPECT_EQ(invoke(index), NVAPI_NOT_SUPPORTED);
    }
    EXPECT_EQ(state.loads, 0u);
    EXPECT_EQ(state.proc_lookups, 0u);
    EXPECT_EQ(state.frees, 0u);
    EXPECT_TRUE(state.resolved_ids.empty());
    EXPECT_TRUE(state.calls.empty());
  }

  TEST_F(NvapiWrapperTest, RepeatedInitializeReusesTheModuleAndRejectsLoaderReplacementWhileLoaded) {
    ASSERT_EQ(NvAPI_Initialize(), NVAPI_OK);
    const auto resolved = state.resolved_ids;
    EXPECT_FALSE(nvprefs::detail::set_nvapi_loader_for_test(fake_loader));
    EXPECT_FALSE(nvprefs::detail::reset_nvapi_loader_for_test());
    EXPECT_EQ(NvAPI_Initialize(), NVAPI_OK);
    EXPECT_EQ(state.loads, 1u);
    EXPECT_EQ(state.proc_lookups, 1u);
    EXPECT_EQ(state.resolved_ids, resolved);
    EXPECT_EQ(invoke(3), state.forwarded_status);
  }

  TEST_F(NvapiWrapperTest, UnloadIsIdempotentClearsDispatchAndReinitializationResolvesAgain) {
    ASSERT_EQ(NvAPI_Initialize(), NVAPI_OK);
    EXPECT_EQ(invoke(3), state.forwarded_status);
    ASSERT_EQ(NvAPI_Unload(), NVAPI_OK);
    EXPECT_EQ(state.frees, 1u);
    EXPECT_EQ(NvAPI_Unload(), NVAPI_OK);
    EXPECT_EQ(state.frees, 1u);
    const auto calls = state.calls.size();
    for (std::size_t index = 0; index < expected_interfaces.size(); ++index) {
      EXPECT_EQ(invoke(index), NVAPI_NOT_SUPPORTED);
    }
    EXPECT_EQ(state.calls.size(), calls);
    const auto resolved = state.resolved_ids.size();
    ASSERT_EQ(NvAPI_Initialize(), NVAPI_OK);
    EXPECT_EQ(state.loads, 2u);
    EXPECT_EQ(state.proc_lookups, 2u);
    EXPECT_EQ(state.resolved_ids.size(), resolved * 2);
    EXPECT_EQ(invoke(3), state.forwarded_status);
    EXPECT_EQ(state.calls.size(), calls + 1);
  }

  TEST_F(NvapiWrapperTest, MissingLibraryDoesNotResolveOrFreeAndCanBeRetried) {
    state.library_available = false;
    EXPECT_EQ(NvAPI_Initialize(), NVAPI_LIBRARY_NOT_FOUND);
    EXPECT_EQ(state.loads, 1u);
    EXPECT_EQ(state.proc_lookups, 0u);
    EXPECT_EQ(state.frees, 0u);
    EXPECT_TRUE(state.resolved_ids.empty());
    EXPECT_EQ(invoke(3), NVAPI_NOT_SUPPORTED);
    state.library_available = true;
    EXPECT_EQ(NvAPI_Initialize(), NVAPI_OK);
    EXPECT_EQ(state.loads, 2u);
    EXPECT_EQ(state.proc_lookups, 1u);
    EXPECT_EQ(invoke(3), state.forwarded_status);
  }

  TEST_F(NvapiWrapperTest, MissingQueryExportReleasesTheModuleAndCanBeRetried) {
    state.query_available = false;
    EXPECT_EQ(NvAPI_Initialize(), NVAPI_LIBRARY_NOT_FOUND);
    EXPECT_EQ(state.loads, 1u);
    EXPECT_EQ(state.proc_lookups, 1u);
    EXPECT_EQ(state.frees, 1u);
    EXPECT_TRUE(state.resolved_ids.empty());
    EXPECT_EQ(invoke(3), NVAPI_NOT_SUPPORTED);
    state.query_available = true;
    EXPECT_EQ(NvAPI_Initialize(), NVAPI_OK);
    EXPECT_EQ(state.loads, 2u);
    EXPECT_EQ(state.proc_lookups, 2u);
    EXPECT_EQ(invoke(3), state.forwarded_status);
  }

  TEST_F(NvapiWrapperTest, EachMissingFunctionStaysUnsupportedUntilAFreshInitializationResolvesIt) {
    for (std::size_t index = 0; index < expected_interfaces.size(); ++index) {
      SCOPED_TRACE(expected_interfaces[index].name);
      ASSERT_EQ(NvAPI_Unload(), NVAPI_OK);
      state.missing_function = expected_interfaces[index].id;
      ASSERT_EQ(NvAPI_Initialize(), NVAPI_OK);
      const auto calls = state.calls.size();
      EXPECT_EQ(invoke(index), NVAPI_NOT_SUPPORTED);
      EXPECT_EQ(state.calls.size(), calls);
      EXPECT_EQ(invoke((index + 1) % expected_interfaces.size()), state.forwarded_status);
      state.missing_function.reset();
      // Already-loaded initialization must not silently replace an unavailable function.
      EXPECT_EQ(NvAPI_Initialize(), NVAPI_OK);
      EXPECT_EQ(invoke(index), NVAPI_NOT_SUPPORTED);
      ASSERT_EQ(NvAPI_Unload(), NVAPI_OK);
      ASSERT_EQ(NvAPI_Initialize(), NVAPI_OK);
      EXPECT_EQ(invoke(index), state.forwarded_status);
    }
  }

  TEST_F(NvapiWrapperTest, RejectsIncompleteLoaderHooksWithoutReplacingTheInstalledFake) {
    auto incomplete = fake_loader;
    incomplete.load_library = nullptr;
    EXPECT_FALSE(nvprefs::detail::set_nvapi_loader_for_test(incomplete));
    incomplete = fake_loader;
    incomplete.get_proc_address = nullptr;
    EXPECT_FALSE(nvprefs::detail::set_nvapi_loader_for_test(incomplete));
    incomplete = fake_loader;
    incomplete.free_library = nullptr;
    EXPECT_FALSE(nvprefs::detail::set_nvapi_loader_for_test(incomplete));
    EXPECT_EQ(NvAPI_Initialize(), NVAPI_OK);
    EXPECT_EQ(state.loads, 1u);
    EXPECT_EQ(state.proc_lookups, 1u);
    EXPECT_EQ(invoke(3), state.forwarded_status);
  }
}  // namespace

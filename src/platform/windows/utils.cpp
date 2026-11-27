#include "utils.h"

#include "display_config.h"

#include <sstream>
#include <system_error>
#include <vector>

#include "src/logging.h"
#include "src/utility.h"

#include <windows.h>
#include <wtsapi32.h>

namespace {
  std::string utf16ToAcp(const std::wstring &utf16Str) {
	const auto acp = GetACP();

	int codepageLen = WideCharToMultiByte(acp, 0, utf16Str.c_str(), utf16Str.size(), NULL, 0, NULL, NULL);
	if (codepageLen == 0) {
		return "";
	}

	std::string codepageStr(codepageLen, '\0');
	WideCharToMultiByte(acp, 0, utf16Str.c_str(), utf16Str.size(), &codepageStr[0], codepageLen, NULL, NULL);

	return codepageStr;
  }
}  // namespace

std::string utf8ToAcp(const std::string& utf8Str) {
	if (GetACP() == CP_UTF8) {
		return std::string(utf8Str);
	}

	int utf16Len = MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), utf8Str.size(), NULL, 0);
	if (utf16Len == 0) {
		return std::string(utf8Str);
	}

	std::wstring utf16Str(utf16Len, L'\0');
	MultiByteToWideChar(CP_UTF8, 0, utf8Str.c_str(), utf8Str.size(), &utf16Str[0], utf16Len);

	return utf16ToAcp(utf16Str);
}

std::string currentCodePageToCharset() {
  switch (GetACP()) {
    case 65001:
      return "UTF-8";
    case 1200:
      return "UTF-16LE";
    case 1201:
      return "UTF-16BE";
    case 1250:
      return "windows-1250";
    case 1251:
      return "windows-1251";
    case 1252:
      return "windows-1252";
    case 1253:
      return "windows-1253";
    case 1254:
      return "windows-1254";
    case 1255:
      return "windows-1255";
    case 1256:
      return "windows-1256";
    case 1257:
      return "windows-1257";
    case 1258:
      return "windows-1258";
    case 936:
      return "GBK";
    case 949:
      return "EUC-KR";
    case 950:
      return "Big5";
    case 932:
      return "Shift_JIS";
    default:
      return "ISO-8859-1";
  }
}

namespace {
  // Modified from https://github.com/FrogTheFrog/Sunshine/blob/b6f8573d35eff7c55da6965dfa317dc9722bd4ef/src/platform/windows/display_device/windows_utils.cpp
  std::string get_error_string(const LONG error_code) {
	std::stringstream error;
	error << "[code: ";
	switch (error_code) {
		case ERROR_INVALID_PARAMETER:
			error << "ERROR_INVALID_PARAMETER";
			break;
		case ERROR_NOT_SUPPORTED:
			error << "ERROR_NOT_SUPPORTED";
			break;
		case ERROR_ACCESS_DENIED:
			error << "ERROR_ACCESS_DENIED";
			break;
		case ERROR_INSUFFICIENT_BUFFER:
			error << "ERROR_INSUFFICIENT_BUFFER";
			break;
		case ERROR_GEN_FAILURE:
			error << "ERROR_GEN_FAILURE";
			break;
		case ERROR_SUCCESS:
			error << "ERROR_SUCCESS";
			break;
		default:
			error << error_code;
			break;
	}
	error << ", message: " << std::system_category().message(static_cast<int>(error_code)) << "]";
	return error.str();
  }

  bool test_no_access_to_ccd_api() {
    platf::display_config::topology_t topology;
    // The former loop retried this race without sleeping or an upper bound. Eight immediate
    // attempts retain its transient-topology tolerance without permitting an infinite stall.
    const auto query = platf::display_config::query_topology(
      QDC_ONLY_ACTIVE_PATHS | QDC_VIRTUAL_MODE_AWARE,
      topology,
      {8, platf::display_config::retry_delay_e::none}
    );
    if (!query) {
      BOOST_LOG(error) << get_error_string(query.status) << " failed to query display paths and modes!";
      BOOST_LOG(debug) << "test_no_access_to_ccd_api failed to query the display topology.";
      return true;
    }

	// Here we are supplying the retrieved display data back to SetDisplayConfig (with VALIDATE flag only, so that we make no actual changes).
	// Unless something is really broken on Windows, this call should never fail under normal circumstances - the configuration is 100% correct, since it was
	// provided by Windows.
	const LONG result = platf::display_config::set_display_config(topology, SDC_VALIDATE | SDC_USE_SUPPLIED_DISPLAY_CONFIG | SDC_VIRTUAL_MODE_AWARE);

	BOOST_LOG(debug) << "test_no_access_to_ccd_api result: " << get_error_string(result);
	return result == ERROR_ACCESS_DENIED;
  }
}  // namespace

std::optional<bool> query_console_session_locked() {
  LPWSTR buffer = nullptr;
  const auto cleanup_guard = util::fail_guard([&buffer]() {
    if (buffer) {
      WTSFreeMemory(buffer);
    }
  });
  DWORD buffer_size_in_bytes = 0;
  if (!WTSQuerySessionInformationW(WTS_CURRENT_SERVER_HANDLE, WTSGetActiveConsoleSessionId(), WTSSessionInfoEx, &buffer, &buffer_size_in_bytes)) {
    BOOST_LOG(error) << get_error_string(GetLastError()) << " failed while calling WTSQuerySessionInformationW!";
    return std::nullopt;
  }
  const auto *wts_info = reinterpret_cast<const WTSINFOEXW *>(buffer);
  if (!wts_info || buffer_size_in_bytes < sizeof(WTSINFOEXW) || wts_info->Level != 1) {
    BOOST_LOG(warning) << "Failed to get valid console session info in query_console_session_locked.";
    return std::nullopt;
  }
  const auto flags = wts_info->Data.WTSInfoExLevel1.SessionFlags;
  if (flags != WTS_SESSIONSTATE_LOCK && flags != WTS_SESSIONSTATE_UNLOCK) {
    BOOST_LOG(warning) << "Unknown console session lock state in query_console_session_locked: " << flags;
    return std::nullopt;
  }
  const bool locked = flags == WTS_SESSIONSTATE_LOCK;
  BOOST_LOG(debug) << "query_console_session_locked: " << locked;
  return locked;
}

bool is_changing_settings_going_to_fail() {
	return query_console_session_locked().value_or(false) || test_no_access_to_ccd_api();
}

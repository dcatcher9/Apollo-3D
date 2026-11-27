/**
 * @file src/platform/windows/misc.cpp
 * @brief Miscellaneous definitions for Windows.
 */
// standard includes
#include <filesystem>
#include <string>

// prevent clang format from "optimizing" the header include order
// clang-format off
#include <Windows.h>
#include <WinUser.h>
// clang-format on

// local includes
#include "misc.h"
#include "src/logging.h"
#include "src/platform/common.h"
#include "src/utility.h"

static std::string ensureCrLf(const std::string& utf8Str);
static std::wstring getClipboardData();
static int setClipboardData(const std::wstring& utf16Str);

using namespace std::literals;

namespace platf {
  std::filesystem::path appdata() {
    WCHAR sunshine_path[MAX_PATH];
    GetModuleFileNameW(nullptr, sunshine_path, _countof(sunshine_path));
    return std::filesystem::path {sunshine_path}.remove_filename() / L"config"sv;
  }

  bool syncThreadDesktop() {
    // SetThreadDesktop keeps the supplied desktop handle in use for the lifetime of the thread's
    // association. Retain the successful handle instead of immediately calling CloseDesktop(),
    // which is documented to fail for a handle used by a thread in this process.
    static thread_local HDESK active_input_desktop = nullptr;

    struct desktop_operations_t {
      HDESK open_input_desktop() {
        const auto desktop = OpenInputDesktop(DF_ALLOWOTHERACCOUNTHOOK, FALSE, GENERIC_ALL);
        if (!desktop) {
          const auto err = GetLastError();
          BOOST_LOG(error) << "Failed to Open Input Desktop [0x"sv << util::hex(err).to_string_view() << ']';
        }
        return desktop;
      }

      bool set_thread_desktop(HDESK desktop) {
        if (SetThreadDesktop(desktop)) {
          return true;
        }
        const auto err = GetLastError();
        BOOST_LOG(error) << "Failed to sync desktop to thread [0x"sv << util::hex(err).to_string_view() << ']';
        return false;
      }

      void close_desktop(HDESK desktop) {
        CloseDesktop(desktop);
      }
    } operations;

    // Once the thread is associated through the new handle, the previous handle is no longer the
    // one in use and can be closed. The final retained handle is released by the process at exit.
    return detail::replace_thread_desktop_handle(active_input_desktop, operations);
  }

  void print_status(const std::string_view &prefix, HRESULT status) {
    char err_string[1024];

    DWORD bytes = FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, status, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), err_string, sizeof(err_string), nullptr);

    BOOST_LOG(error) << prefix << ": "sv << std::string_view {err_string, bytes};
  }

  std::wstring from_utf8(const std::string_view &string) {
    // No conversion needed if the string is empty
    if (string.empty()) {
      return {};
    }

    // Get the output size required to store the string
    auto output_size = MultiByteToWideChar(CP_UTF8, 0, string.data(), string.size(), nullptr, 0);
    if (output_size == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(error) << "Failed to get UTF-16 buffer size: "sv << winerr;
      return {};
    }

    // Perform the conversion
    std::wstring output(output_size, L'\0');
    output_size = MultiByteToWideChar(CP_UTF8, 0, string.data(), string.size(), output.data(), output.size());
    if (output_size == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(error) << "Failed to convert string to UTF-16: "sv << winerr;
      return {};
    }

    return output;
  }

  std::string to_utf8(const std::wstring_view &string) {
    // No conversion needed if the string is empty
    if (string.empty()) {
      return {};
    }

    // Get the output size required to store the string
    auto output_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, string.data(), string.size(), nullptr, 0, nullptr, nullptr);
    if (output_size == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(error) << "Failed to get UTF-8 buffer size: "sv << winerr;
      return {};
    }

    // Perform the conversion
    std::string output(output_size, '\0');
    output_size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, string.data(), string.size(), output.data(), output.size(), nullptr, nullptr);
    if (output_size == 0) {
      auto winerr = GetLastError();
      BOOST_LOG(error) << "Failed to convert string to UTF-8: "sv << winerr;
      return {};
    }

    return output;
  }

  std::uint32_t windows_build_number() {
    // GetVersionEx answers for the version the executable's manifest names; this one does not.
    using rtl_get_version_fn = LONG(WINAPI *)(OSVERSIONINFOW *);
    const auto ntdll = GetModuleHandleW(L"ntdll.dll");
    const auto rtl_get_version = ntdll ?
                                   reinterpret_cast<rtl_get_version_fn>(reinterpret_cast<void *>(GetProcAddress(ntdll, "RtlGetVersion"))) :
                                   nullptr;
    OSVERSIONINFOW version {};
    version.dwOSVersionInfoSize = sizeof(version);
    if (!rtl_get_version || rtl_get_version(&version) != 0) {
      return 0;
    }
    return version.dwBuildNumber;
  }

  std::string
  get_clipboard() {
    std::string currentClipboard = to_utf8(getClipboardData());
    return currentClipboard;
  }

  bool
  set_clipboard(const std::string& content) {
    std::wstring cpContent = from_utf8(ensureCrLf(content));
    return !setClipboardData(cpContent);
  }
}  // namespace platf

static std::string ensureCrLf(const std::string& utf8Str) {
    std::string result;
    result.reserve(utf8Str.size() + utf8Str.size() / 2); // Reserve extra space

    for (size_t i = 0; i < utf8Str.size(); ++i) {
        if (utf8Str[i] == '\n' && (i == 0 || utf8Str[i - 1] != '\r')) {
            result += '\r'; // Add \r before \n if not present
        }
        result += utf8Str[i]; // Always add the current character
    }

    return result;
}

static std::wstring getClipboardData() {
  if (!OpenClipboard(nullptr)) {
    BOOST_LOG(warning) << "Failed to open clipboard.";
    return L"";
  }

  HANDLE hData = GetClipboardData(CF_UNICODETEXT);
  if (hData == nullptr) {
    BOOST_LOG(warning) << "No text data in clipboard or failed to get data.";
    CloseClipboard();
    return L"";
  }

  wchar_t* pszText = static_cast<wchar_t*>(GlobalLock(hData));
  if (pszText == nullptr) {
    BOOST_LOG(warning) << "Failed to lock clipboard data.";
    CloseClipboard();
    return L"";
  }

  std::wstring ret = pszText;

  GlobalUnlock(hData);
  CloseClipboard();

  return ret;
}

static int setClipboardData(const std::wstring& utf16Str) {
  if (!OpenClipboard(nullptr)) {
    BOOST_LOG(warning) << "Failed to open clipboard.";
    return 1;
  }

  if (!EmptyClipboard()) {
    BOOST_LOG(warning) << "Failed to empty clipboard.";
    CloseClipboard();
    return 1;
  }

  // Allocate global memory for the clipboard text
  HGLOBAL hGlobal = GlobalAlloc(GMEM_MOVEABLE, utf16Str.size() * 2 + 2);
  if (hGlobal == nullptr) {
    BOOST_LOG(warning) << "Failed to allocate global memory.";
    CloseClipboard();
    return 1;
  }

  // Lock the global memory and copy the text
  char* pGlobal = static_cast<char*>(GlobalLock(hGlobal));
  if (pGlobal == nullptr) {
    BOOST_LOG(warning) << "Failed to lock global memory.";
    GlobalFree(hGlobal);
    CloseClipboard();
    return 1;
  }

  memcpy(pGlobal, utf16Str.c_str(), utf16Str.size() * 2 + 2);
  GlobalUnlock(hGlobal);

  // Set the clipboard data
  if (SetClipboardData(CF_UNICODETEXT, hGlobal) == nullptr) {
    BOOST_LOG(warning) << "Failed to set clipboard data.";
    GlobalFree(hGlobal);
    CloseClipboard();
    return 1;
  }

  CloseClipboard();

  return 0;
}

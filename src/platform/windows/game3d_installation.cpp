// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_installation.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <fstream>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <Windows.h>

#include <ShlObj.h>
#include <TlHelp32.h>

namespace platf::game3d {
  namespace {
    namespace fs = std::filesystem;
    std::atomic_bool installation_cancelled {false};

    struct handle_t {
      HANDLE value {INVALID_HANDLE_VALUE};

      ~handle_t() {
        if (value != INVALID_HANDLE_VALUE && value != nullptr) {
          CloseHandle(value);
        }
      }
    };

    std::string utf8(const fs::path &path) {
      const auto value = path.u8string();
      return {value.begin(), value.end()};
    }

    std::string trim(std::string value) {
      const auto first = value.find_first_not_of(" \t\r\n");
      return first == std::string::npos ? std::string {} : value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
    }

    // ReShade uses a single comma as an array separator and doubled commas as a
    // literal comma. AddonPath is a scalar; ambiguous arrays cannot select a target.
    std::optional<fs::path> configured_addon_directory(const fs::path &ini) {
      std::ifstream input(ini, std::ios::binary);
      if (!input) {
        return std::nullopt;
      }
      std::string section;
      std::string line;
      std::string value = ".";
      bool found_section = false;
      bool found_key = false;
      std::size_t bytes = 0;
      while (std::getline(input, line)) {
        bytes += line.size();
        if (bytes > 1024 * 1024) {
          return std::nullopt;
        }
        if (bytes == line.size() && line.starts_with("\xEF\xBB\xBF")) {
          line.erase(0, 3);
        }
        line = trim(std::move(line));
        if (line.empty() || line[0] == ';' || line[0] == '#' || line[0] == '/') {
          continue;
        }
        if (line[0] == '[') {
          const auto close = line.find(']');
          if (close == std::string::npos) {
            return std::nullopt;
          }
          section = trim(line.substr(1, close - 1));
          if (section == "ADDON") {
            if (found_section) {
              return std::nullopt;
            }
            found_section = true;
          }
          continue;
        }
        const auto equals = line.find('=');
        if (section == "ADDON" && equals != std::string::npos && trim(line.substr(0, equals)) == "AddonPath") {
          if (found_key) {
            return std::nullopt;
          }
          found_key = true;
          const auto encoded = trim(line.substr(equals + 1));
          value.clear();
          for (std::size_t i = 0; i < encoded.size(); ++i) {
            if (encoded[i] == ',') {
              if (i + 1 == encoded.size() || encoded[i + 1] != ',') {
                return std::nullopt;
              }
              ++i;
            }
            value.push_back(encoded[i]);
          }
          if (value.empty()) {
            value = ".";
          }
        }
      }
      if (!input.eof()) {
        return std::nullopt;
      }
      // The game expands environment variables in its own user's environment. The setup worker
      // runs as the host's account (SYSTEM for the service), where %LOCALAPPDATA% and similar
      // resolve to another profile; such a path needs manual setup rather than a guess.
      if (value.find('%') != std::string::npos) {
        return std::nullopt;
      }
      const auto path = fs::path(std::u8string(value.begin(), value.end()));
      return (path.is_absolute() ? path : ini.parent_path() / path).lexically_normal();
    }

    bool regular_file(const fs::path &path) {
      std::error_code error;
      return fs::is_regular_file(path, error);
    }

    bool has_reparse_component(const fs::path &path) {
      fs::path prefix;
      for (const auto &part : path) {
        prefix /= part;
        const auto attributes = GetFileAttributesW(prefix.c_str());
        if (attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
          return true;
        }
      }
      return false;
    }

    bool same_path(const fs::path &a, const fs::path &b) {
      return CompareStringOrdinal(a.c_str(), -1, b.c_str(), -1, TRUE) == CSTR_EQUAL;
    }

    bool starts_with_name(std::wstring_view name, std::wstring_view prefix) {
      return name.size() >= prefix.size() && CompareStringOrdinal(name.data(), static_cast<int>(prefix.size()), prefix.data(), static_cast<int>(prefix.size()), TRUE) == CSTR_EQUAL;
    }

    bool ends_with_name(std::wstring_view name, std::wstring_view suffix) {
      return name.size() >= suffix.size() && CompareStringOrdinal(name.data() + name.size() - suffix.size(), static_cast<int>(suffix.size()), suffix.data(), static_cast<int>(suffix.size()), TRUE) == CSTR_EQUAL;
    }

    // Files and folders that kernel-level anti-cheat systems install inside a game's folder.
    // tools/reshade/install-runtime.ps1 applies the same markers and walk independently.
    bool anti_cheat_marker(std::wstring_view name) {
      for (const std::wstring_view prefix : {L"EasyAntiCheat", L"BattlEye", L"EAAntiCheat", L"XIGNCODE", L"AntiCheatExpert", L"HoYoKProtect", L"mhyprot"}) {
        if (starts_with_name(name, prefix)) {
          return true;
        }
      }
      for (const std::wstring_view exact : {L"start_protected_game.exe", L"GameGuard", L"randgrid.sys"}) {
        if (name.size() == exact.size() && starts_with_name(name, exact)) {
          return true;
        }
      }
      return ends_with_name(name, L"_BE.exe");
    }

    // Program Files and the Windows directory hold the anti-cheat services' own system-wide
    // installations, and a volume root holds unrelated libraries; markers there do not
    // identify the game, so the walk stops before them.
    bool shared_root(const fs::path &directory) {
      if (!directory.has_relative_path()) {
        return true;
      }
      for (const auto &folder : {FOLDERID_ProgramFilesX64, FOLDERID_ProgramFilesX86, FOLDERID_Windows}) {
        PWSTR known = nullptr;
        const auto result = SHGetKnownFolderPath(folder, KF_FLAG_DEFAULT, nullptr, &known);
        const bool same = SUCCEEDED(result) && known && same_path(fs::path(known).lexically_normal(), directory);
        CoTaskMemFree(known);
        if (same) {
          return true;
        }
      }
      return false;
    }

    enum class anti_cheat_e {
      absent,
      present,
      unknown,
    };

    // Checks the executable's folder and up to four parents: the game root of Unreal-style
    // <game>/<project>/Binaries/Win64 layouts, or at most its library folder. Sibling games'
    // folders are never entered. A folder that cannot be listed completely is unknown.
    anti_cheat_e find_anti_cheat(const fs::path &executable, fs::path &marker) {
      auto directory = executable.parent_path();
      for (int level = 0; level < 5 && !shared_root(directory); ++level) {
        WIN32_FIND_DATAW entry {};
        const auto pattern = directory / L"*";
        const auto find = FindFirstFileExW(pattern.c_str(), FindExInfoBasic, &entry, FindExSearchNameMatch, nullptr, FIND_FIRST_EX_LARGE_FETCH);
        if (find == INVALID_HANDLE_VALUE) {
          return anti_cheat_e::unknown;
        }
        bool complete = false;
        for (std::size_t entries = 0; entries < 65536; ++entries) {
          if (anti_cheat_marker(entry.cFileName)) {
            marker = directory / entry.cFileName;
            FindClose(find);
            return anti_cheat_e::present;
          }
          if (!FindNextFileW(find, &entry)) {
            complete = GetLastError() == ERROR_NO_MORE_FILES;
            break;
          }
        }
        FindClose(find);
        if (!complete) {
          return anti_cheat_e::unknown;
        }
        directory = directory.parent_path();
      }
      return anti_cheat_e::absent;
    }

    bool reshade_runtime_present(const fs::path &directory) {
      for (const auto *name : {L"dxgi.dll", L"d3d11.dll", L"d3d12.dll", L"d3d10.dll", L"d3d10_1.dll", L"d3d9.dll", L"opengl32.dll", L"dinput8.dll"}) {
        const auto candidate = directory / name;
        const auto size = GetFileVersionInfoSizeW(candidate.c_str(), nullptr);
        if (!size || size > 1024 * 1024) {
          continue;
        }
        std::vector<unsigned char> version(size);
        if (!GetFileVersionInfoW(candidate.c_str(), 0, size, version.data())) {
          continue;
        }

        struct translation_t {
          WORD language;
          WORD codepage;
        };

        translation_t *translations = nullptr;
        UINT bytes = 0;
        if (!VerQueryValueW(version.data(), L"\\VarFileInfo\\Translation", reinterpret_cast<void **>(&translations), &bytes)) {
          continue;
        }
        for (std::size_t i = 0; i < bytes / sizeof(translation_t); ++i) {
          std::array<wchar_t, 96> key {};
          swprintf(key.data(), key.size(), L"\\StringFileInfo\\%04x%04x\\ProductName", translations[i].language, translations[i].codepage);
          wchar_t *product = nullptr;
          UINT characters = 0;
          if (VerQueryValueW(version.data(), key.data(), reinterpret_cast<void **>(&product), &characters) && characters == 8 && CompareStringOrdinal(product, 7, L"ReShade", 7, TRUE) == CSTR_EQUAL) {
            return true;
          }
        }
      }
      return false;
    }

    // Unknown process identity is treated like a running game. Inaccessible
    // unrelated names do not block updates; inaccessible matching names do.
    bool game_running_or_unknown(const fs::path &executable) {
      handle_t snapshot {CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)};
      if (snapshot.value == INVALID_HANDLE_VALUE) {
        return true;
      }
      PROCESSENTRY32W entry {};
      entry.dwSize = sizeof(entry);
      if (!Process32FirstW(snapshot.value, &entry)) {
        return true;
      }
      do {
        if (!same_path(fs::path(entry.szExeFile), executable.filename())) {
          continue;
        }
        handle_t process {OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ProcessID)};
        if (!process.value) {
          // A process may have exited between the snapshot and OpenProcess.
          if (GetLastError() == ERROR_INVALID_PARAMETER) {
            continue;
          }
          return true;
        }
        std::wstring path(32768, L'\0');
        DWORD length = static_cast<DWORD>(path.size());
        if (!QueryFullProcessImageNameW(process.value, 0, path.data(), &length)) {
          return true;
        }
        path.resize(length);
        if (same_path(fs::path(path).lexically_normal(), executable)) {
          return true;
        }
      } while (Process32NextW(snapshot.value, &entry));
      return GetLastError() != ERROR_NO_MORE_FILES;
    }

    std::optional<bool> identical_files(const fs::path &a, const fs::path &b) {
      std::error_code error;
      const auto a_size = fs::file_size(a, error);
      if (error || a_size > 128 * 1024 * 1024) {
        return std::nullopt;
      }
      const auto b_size = fs::file_size(b, error);
      if (error || b_size > 128 * 1024 * 1024) {
        return std::nullopt;
      }
      if (a_size != b_size) {
        return false;
      }
      std::ifstream left(a, std::ios::binary);
      std::ifstream right(b, std::ios::binary);
      if (!left || !right) {
        return std::nullopt;
      }
      std::array<char, 65536> l {};
      std::array<char, 65536> r {};
      while (left && right) {
        left.read(l.data(), l.size());
        right.read(r.data(), r.size());
        if (left.gcount() != right.gcount() || !std::equal(l.begin(), l.begin() + left.gcount(), r.begin())) {
          return false;
        }
      }
      return left.eof() && right.eof() ? std::optional<bool>(true) : std::nullopt;
    }

    installation_status unavailable(installation_status status, std::string detail) {
      status.state = installation_state_e::unavailable;
      status.detail = std::move(detail);
      return status;
    }

    struct staged_file_t {
      fs::path path;

      ~staged_file_t() {
        if (!path.empty()) {
          DeleteFileW(path.c_str());
        }
      }

      bool copy(const fs::path &source, const fs::path &directory) {
        std::array<wchar_t, MAX_PATH> name {};
        if (!GetTempFileNameW(directory.c_str(), L"s3d", 0, name.data())) {
          return false;
        }
        path = name.data();
        if (!CopyFileW(source.c_str(), path.c_str(), FALSE)) {
          return false;
        }
        return identical_files(source, path) == true;
      }
    };

    std::wstring quoted_argument(std::wstring_view value) {
      std::wstring result = L"\"";
      std::size_t slashes = 0;
      for (const auto character : value) {
        if (character == L'\\') {
          ++slashes;
          continue;
        }
        result.append(slashes * (character == L'\"' ? 2 : 1), L'\\');
        slashes = 0;
        if (character == L'\"') {
          result += L'\\';
        }
        result += character;
      }
      result.append(slashes * 2, L'\\');
      result += L'\"';
      return result;
    }

    std::string valid_utf8(const std::string &input) {
      if (input.empty()) {
        return {};
      }
      // Flags zero replaces malformed or truncated input with U+FFFD. Keep
      // captured process output safe for the Web UI's UTF-8 JSON serializer.
      const auto characters = MultiByteToWideChar(CP_UTF8, 0, input.data(), input.size(), nullptr, 0);
      if (!characters) {
        return "Installer output was unavailable.";
      }
      std::wstring wide(characters, L'\0');
      MultiByteToWideChar(CP_UTF8, 0, input.data(), input.size(), wide.data(), wide.size());
      const auto bytes = WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide.size(), nullptr, 0, nullptr, nullptr);
      std::string result(bytes, '\0');
      WideCharToMultiByte(CP_UTF8, 0, wide.data(), wide.size(), result.data(), result.size(), nullptr, nullptr);
      return result;
    }

    std::optional<std::string> run_installer(const fs::path &script, const std::wstring &arguments, DWORD timeout_ms) {
      if (installation_cancelled.load()) {
        return "Installation was cancelled because the host is shutting down.";
      }
      std::array<wchar_t, MAX_PATH> system_directory {};
      const auto length = GetSystemDirectoryW(system_directory.data(), system_directory.size());
      if (!length || length >= system_directory.size()) {
        return "Windows PowerShell could not be located.";
      }
      const auto powershell = fs::path(system_directory.data()) / "WindowsPowerShell" / "v1.0" / "powershell.exe";
      // A host launched from PowerShell 7 can inherit its module path. Windows
      // PowerShell 5 must load its own modules (not bundled PowerShell 7 modules).
      auto inherited_environment = GetEnvironmentStringsW();
      if (!inherited_environment) {
        return "The installer environment could not be initialized.";
      }
      std::vector<std::wstring> environment_entries;
      for (auto entry = inherited_environment; *entry; entry += wcslen(entry) + 1) {
        if (wcslen(entry) < 13 || CompareStringOrdinal(entry, 13, L"PSModulePath=", 13, TRUE) != CSTR_EQUAL) {
          environment_entries.emplace_back(entry);
        }
      }
      FreeEnvironmentStringsW(inherited_environment);
      environment_entries.push_back(L"PSModulePath=" + (powershell.parent_path() / "Modules").wstring());
      std::sort(environment_entries.begin(), environment_entries.end(), [](const auto &a, const auto &b) {
        return CompareStringOrdinal(a.c_str(), -1, b.c_str(), -1, TRUE) == CSTR_LESS_THAN;
      });
      std::vector<wchar_t> environment;
      for (const auto &entry : environment_entries) {
        environment.insert(environment.end(), entry.begin(), entry.end());
        environment.push_back(L'\0');
      }
      environment.push_back(L'\0');
      auto command = quoted_argument(powershell.wstring()) +
                     L" -NoLogo -NoProfile -NonInteractive -OutputFormat Text -ExecutionPolicy Bypass -File " + quoted_argument(script.wstring()) + L" " + arguments;
      handle_t job {CreateJobObjectW(nullptr, nullptr)};
      JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
      limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      if (!job.value || !SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        return "The installer could not be started safely.";
      }
      SECURITY_ATTRIBUTES security {sizeof(security), nullptr, TRUE};
      handle_t output_read;
      handle_t output_write;
      if (!CreatePipe(&output_read.value, &output_write.value, &security, 0) || !SetHandleInformation(output_read.value, HANDLE_FLAG_INHERIT, 0)) {
        return "The installer output could not be captured.";
      }
      handle_t input {CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)};
      if (input.value == INVALID_HANDLE_VALUE) {
        return "The installer input could not be initialized.";
      }
      SIZE_T attribute_bytes = 0;
      InitializeProcThreadAttributeList(nullptr, 1, 0, &attribute_bytes);
      std::vector<unsigned char> attributes(attribute_bytes);
      auto attribute_list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
      if (!InitializeProcThreadAttributeList(attribute_list, 1, 0, &attribute_bytes)) {
        return "The installer handle list could not be initialized.";
      }

      struct attribute_cleanup_t {
        LPPROC_THREAD_ATTRIBUTE_LIST value;

        ~attribute_cleanup_t() {
          DeleteProcThreadAttributeList(value);
        }
      } attribute_cleanup {attribute_list};

      HANDLE inherited[] {input.value, output_write.value};
      if (!UpdateProcThreadAttribute(attribute_list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof(inherited), nullptr, nullptr)) {
        return "The installer handle list could not be configured.";
      }
      STARTUPINFOEXW startup {};
      startup.StartupInfo.cb = sizeof(startup);
      startup.StartupInfo.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
      startup.StartupInfo.wShowWindow = SW_HIDE;
      startup.StartupInfo.hStdInput = input.value;
      startup.StartupInfo.hStdOutput = output_write.value;
      startup.StartupInfo.hStdError = output_write.value;
      startup.lpAttributeList = attribute_list;
      PROCESS_INFORMATION created {};
      if (!CreateProcessW(powershell.c_str(), command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT, environment.data(), script.parent_path().c_str(), &startup.StartupInfo, &created)) {
        return "The bundled installer could not be started.";
      }
      handle_t process {created.hProcess};
      handle_t thread {created.hThread};
      if (!AssignProcessToJobObject(job.value, process.value) || ResumeThread(thread.value) == static_cast<DWORD>(-1)) {
        TerminateProcess(process.value, 1);
        return "The bundled installer could not be started safely.";
      }
      std::string output;
      const auto deadline = GetTickCount64() + timeout_ms;
      const auto drain_output = [&] {
        DWORD available = 0;
        while (PeekNamedPipe(output_read.value, nullptr, 0, nullptr, &available, nullptr) && available) {
          std::array<char, 4096> bytes {};
          DWORD read = 0;
          if (!ReadFile(output_read.value, bytes.data(), std::min<DWORD>(available, bytes.size()), &read, nullptr) || !read) {
            break;
          }
          if (output.size() < 4096) {
            output.append(bytes.data(), std::min<std::size_t>(read, 4096 - output.size()));
          }
        }
      };
      DWORD wait_result;
      do {
        if (installation_cancelled.load()) {
          TerminateJobObject(job.value, 1);
          return "Installation was cancelled because the host is shutting down.";
        }
        drain_output();
        wait_result = WaitForSingleObject(process.value, 50);
      } while (wait_result == WAIT_TIMEOUT && GetTickCount64() < deadline);
      if (wait_result != WAIT_OBJECT_0) {
        TerminateJobObject(job.value, 1);
        return "The installer timed out. Inspect the game's Sunshine3D-backups folder before retrying manually.";
      }
      drain_output();
      DWORD exit_code = 1;
      if (!GetExitCodeProcess(process.value, &exit_code) || exit_code != 0) {
        return "The bundled installer failed. " + trim(valid_utf8(output));
      }
      return std::nullopt;
    }
  }  // namespace

  std::filesystem::path bundled_addon_directory() {
    std::wstring path(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (!length || length == path.size()) {
      return {};
    }
    path.resize(length);
    return fs::path(path).parent_path() / "tools" / "reshade";
  }

  void cancel_installation_work() {
    installation_cancelled.store(true);
  }

  std::string file_identity(const fs::path &path) {
    handle_t file {CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)};
    FILE_ID_INFO id {};
    FILE_BASIC_INFO basic {};
    FILE_STANDARD_INFO standard {};
    if (file.value == INVALID_HANDLE_VALUE || !GetFileInformationByHandleEx(file.value, FileIdInfo, &id, sizeof(id)) || !GetFileInformationByHandleEx(file.value, FileBasicInfo, &basic, sizeof(basic)) || !GetFileInformationByHandleEx(file.value, FileStandardInfo, &standard, sizeof(standard)) || standard.Directory) {
      return {};
    }
    std::string result;
    const auto append = [&result](const void *data, std::size_t bytes) {
      constexpr char digits[] = "0123456789abcdef";
      for (const auto byte : std::span {static_cast<const unsigned char *>(data), bytes}) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
      }
    };
    append(&id.VolumeSerialNumber, sizeof(id.VolumeSerialNumber));
    append(id.FileId.Identifier, sizeof(id.FileId.Identifier));
    append(&standard.EndOfFile.QuadPart, sizeof(standard.EndOfFile.QuadPart));
    append(&basic.LastWriteTime.QuadPart, sizeof(basic.LastWriteTime.QuadPart));
    return result;
  }

  installation_status inspect_installation(const fs::path &game_executable, const fs::path &bundle_directory) {
    installation_status status;
    try {
      std::error_code error;
      status.executable = fs::weakly_canonical(game_executable, error);
      if (error || !regular_file(status.executable) || !same_path(status.executable.extension(), L".exe")) {
        return unavailable(std::move(status), "The game executable is unavailable.");
      }
      fs::path marker;
      switch (find_anti_cheat(status.executable, marker)) {
        case anti_cheat_e::present:
          status.state = installation_state_e::anti_cheat;
          status.detail = "Anti-cheat detected (" + utf8(marker.filename()) + " in " + utf8(marker.parent_path()) + "), so Sunshine 3D does not offer or perform setup for this game.";
          return status;
        case anti_cheat_e::unknown:
          return unavailable(std::move(status), "The game's folders could not be checked for anti-cheat, so Sunshine 3D does not offer setup for this game.");
        case anti_cheat_e::absent:
          break;
      }
      const auto ini = status.executable.parent_path() / "ReShade.ini";
      status.reshade_present = regular_file(ini) && reshade_runtime_present(status.executable.parent_path());
      if (!status.reshade_present) {
        status.state = installation_state_e::missing_reshade;
        status.detail = "Install ReShade with full add-on support for this game's executable, then install the Sunshine 3D add-on.";
        return status;
      }
      const auto addon_directory = configured_addon_directory(ini);
      if (!addon_directory) {
        return unavailable(std::move(status), "ReShade.ini has an unreadable or ambiguous add-on path. Run the Sunshine 3D installer after correcting it.");
      }
      status.addon = *addon_directory / "SunshineSBS.addon64";
      status.addon_present = regular_file(status.addon);
      if (!status.addon_present) {
        status.state = installation_state_e::missing_addon;
        status.detail = "Install the bundled Sunshine 3D add-on for this executable. The next game launch will load it.";
        return status;
      }
      const auto source = bundle_directory / "SunshineSBS.addon64";
      if (bundle_directory.empty() || !regular_file(source) || !regular_file(bundle_directory / "SunshineSBS-LICENSES.txt")) {
        return unavailable(std::move(status), "The matching Sunshine 3D add-on package is unavailable. Repair or rebuild the host package.");
      }
      const auto identical = identical_files(source, status.addon);
      if (!identical) {
        return unavailable(std::move(status), "The installed add-on could not be compared with the host package.");
      }
      status.state = *identical ? installation_state_e::current : installation_state_e::update_available;
      status.detail = *identical ? "The installed Sunshine 3D add-on matches this host." : "The Sunshine 3D add-on will update automatically when the game is closed.";
      return status;
    } catch (const std::exception &) {
      return unavailable(std::move(status), "The game's ReShade installation could not be inspected.");
    }
  }

  installation_status update_existing_addon(const fs::path &game_executable, const fs::path &bundle_directory) {
    auto status = inspect_installation(game_executable, bundle_directory);
    if (status.state != installation_state_e::update_available) {
      return status;
    }
    const auto pending = [&status](std::string detail) {
      status.state = installation_state_e::update_pending;
      status.detail = std::move(detail);
      return status;
    };
    try {
      const auto directory = status.addon.parent_path();
      const auto backup = fs::path(status.addon.wstring() + L".previous");
      const auto license = directory / "SunshineSBS-LICENSES.txt";
      if (has_reparse_component(status.addon) || has_reparse_component(backup) || has_reparse_component(license)) {
        return unavailable(std::move(status), "Automatic add-on updates do not follow linked installation paths. Run the installer manually.");
      }
      if (game_running_or_unknown(status.executable)) {
        return pending("Keep playing; Sunshine 3D will update its add-on when the game exits, ready for the next launch.");
      }
      staged_file_t staged;
      staged_file_t staged_license;
      if (!staged.copy(bundle_directory / "SunshineSBS.addon64", directory) || !staged_license.copy(bundle_directory / "SunshineSBS-LICENSES.txt", directory)) {
        return pending("The add-on directory is unavailable or not writable. Close the game or run the bundled installer manually.");
      }
      if (game_running_or_unknown(status.executable)) {
        return pending("The game started during its add-on update. Close it to retry automatically.");
      }
      // Check write/delete sharing before the atomic replacement. Loaded or
      // externally locked DLLs are left untouched, even when their owner differs.
      {
        handle_t probe {CreateFileW(status.addon.c_str(), GENERIC_READ | GENERIC_WRITE | DELETE, 0, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)};
        if (probe.value == INVALID_HANDLE_VALUE) {
          return pending("The add-on is in use or cannot be replaced. Close the game to retry automatically.");
        }
      }
      // Publish the accompanying licenses before the DLL, retaining the old DLL
      // with ReplaceFileW's backup. ReShade.ini and presets are never rewritten.
      if (!MoveFileExW(staged_license.path.c_str(), license.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return pending("The add-on licenses could not be updated. Close the game or run the bundled installer manually.");
      }
      if (!ReplaceFileW(status.addon.c_str(), staged.path.c_str(), backup.c_str(), 0, nullptr, nullptr)) {
        return pending("The add-on could not be replaced. Close the game or run the bundled installer manually.");
      }
      status.state = installation_state_e::updated;
      status.detail = "Sunshine 3D updated the installed add-on. Start the game to use the matching version.";
      return status;
    } catch (const std::exception &) {
      return pending("The add-on update could not finish. Close the game or run the bundled installer manually.");
    }
  }

  installation_status install_addon_for_next_launch(const fs::path &game_executable, const fs::path &bundle_directory) {
    auto status = inspect_installation(game_executable, bundle_directory);
    if (status.state != installation_state_e::missing_addon) {
      return status;
    }
    try {
      std::error_code error;
      const auto bundle = fs::weakly_canonical(bundle_directory, error);
      const auto script = bundle / "install.ps1";
      if (error || bundle_directory.empty() || !regular_file(script) || !regular_file(bundle / "SunshineSBS.addon64") || !regular_file(bundle / "SunshineSBS-LICENSES.txt")) {
        return unavailable(std::move(status), "The bundled Sunshine 3D installer is unavailable. Repair or rebuild the host package.");
      }
      if (has_reparse_component(status.executable.parent_path()) || has_reparse_component(status.addon) || has_reparse_component(status.executable.parent_path() / "ReShade.ini") || has_reparse_component(bundle)) {
        return unavailable(std::move(status), "Automatic installation does not follow linked installation paths. Run the installer manually.");
      }
      if (const auto failure = run_installer(script, L"-GameExecutable " + quoted_argument(status.executable.wstring()) + L" -AddonDirectory " + quoted_argument(bundle.wstring()) + L" -AllowRunningGame", 30000)) {
        return unavailable(std::move(status), *failure);
      }
      status = inspect_installation(status.executable, bundle);
      if (status.state != installation_state_e::current) {
        return unavailable(std::move(status), "The installed add-on could not be verified against this host.");
      }
      status.state = installation_state_e::installed;
      status.detail = "Sunshine 3D installed its add-on for the next game launch. Restart the game when ready.";
      return status;
    } catch (const std::exception &) {
      return unavailable(std::move(status), "The add-on installation could not finish. Run the bundled installer manually.");
    }
  }

  installation_status install_reshade_for_next_launch(const fs::path &game_executable, const fs::path &bundle_directory) {
    auto status = inspect_installation(game_executable, bundle_directory);
    if (status.state != installation_state_e::missing_reshade) {
      return status;
    }
    try {
      std::error_code error;
      const auto bundle = fs::weakly_canonical(bundle_directory, error);
      const auto script = bundle / "install-runtime.ps1";
      if (error || bundle_directory.empty() || !regular_file(script)) {
        return unavailable(std::move(status), "The bundled ReShade installer is unavailable. Repair or rebuild the host package.");
      }
      if (has_reparse_component(status.executable.parent_path()) || has_reparse_component(bundle)) {
        return unavailable(std::move(status), "Automatic installation does not follow linked installation paths. Run the installer manually.");
      }
      if (const auto failure = run_installer(script, L"-GameExecutable " + quoted_argument(status.executable.wstring()), 90000)) {
        return unavailable(std::move(status), *failure);
      }
      status = inspect_installation(status.executable, bundle);
      if (!status.reshade_present) {
        return unavailable(std::move(status), "The new ReShade installation could not be verified.");
      }
      status.state = installation_state_e::installed;
      status.detail = "ReShade is installed for the next game launch. Install the Sunshine 3D add-on to finish setup.";
      return status;
    } catch (const std::exception &) {
      return unavailable(std::move(status), "ReShade could not be installed. Run the bundled installer manually.");
    }
  }

  installation_status install_reshade_and_addon_for_next_launch(const fs::path &game_executable, const fs::path &bundle_directory) {
    auto status = inspect_installation(game_executable, bundle_directory);
    if (status.state != installation_state_e::missing_reshade) {
      return status;
    }
    const auto directory = status.executable.parent_path();
    const auto proxy = directory / "dxgi.dll";
    const auto configuration = directory / "ReShade.ini";
    const bool proxy_existed = GetFileAttributesW(proxy.c_str()) != INVALID_FILE_ATTRIBUTES;
    const bool configuration_existed = GetFileAttributesW(configuration.c_str()) != INVALID_FILE_ATTRIBUTES;
    status = install_reshade_for_next_launch(status.executable, bundle_directory);
    if (status.state != installation_state_e::installed) {
      return status;
    }
    // The add-on installer never touches the proxy, but it edits ReShade.ini and restores it
    // from its own backup on failure; compare the proxy by file identity, the INI by content.
    const auto created_proxy = proxy_existed ? std::string {} : file_identity(proxy);
    std::optional<std::string> created_configuration;
    if (!configuration_existed) {
      std::ifstream input(configuration, std::ios::binary);
      created_configuration.emplace(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    status = install_addon_for_next_launch(status.executable, bundle_directory);
    if (status.state == installation_state_e::installed) {
      status.detail = "ReShade and the Sunshine 3D add-on are installed beside the game executable. Keep playing; Game 3D takes effect on the next launch.";
      return status;
    }
    // A bare runtime would show ReShade's own overlay in the game without Game 3D. Remove only
    // the files this call created and nobody changed since, proxy first; the folder is then
    // fresh again for a retry.
    std::string kept;
    const auto keep = [&kept](const fs::path &path) {
      kept += (kept.empty() ? "" : ", ") + utf8(path);
    };
    const auto exists = [](const fs::path &path) {
      return GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES;
    };
    if (!proxy_existed && exists(proxy) && (created_proxy.empty() || file_identity(proxy) != created_proxy || !DeleteFileW(proxy.c_str()))) {
      keep(proxy);
    }
    if (created_configuration && exists(configuration)) {
      std::string current;
      bool read = false;
      if (std::ifstream input {configuration, std::ios::binary}) {
        current.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        read = !input.bad();
      }
      if (!read || current != *created_configuration || !DeleteFileW(configuration.c_str())) {
        keep(configuration);
      }
    }
    const auto failure = status.detail;
    status = inspect_installation(status.executable, bundle_directory);
    if (!kept.empty()) {
      return unavailable(std::move(status), "The Sunshine 3D add-on could not be installed, and the new ReShade files could not be removed again: " + kept + ". Remove them or finish setup manually. " + failure);
    }
    return unavailable(std::move(status), "The Sunshine 3D add-on could not be installed, so the new ReShade runtime was removed again; setup retries on the next game launch. " + failure);
  }
}  // namespace platf::game3d

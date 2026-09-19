// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>
#include <array>
#include <cstdio>
#include <cwchar>
#include <string_view>
#include <vector>

namespace sunshine_streamline::versioning {
  enum class abi { unsupported, v1_1_1, v2_7_30 };
  struct information {
    DWORD file_ms{}, file_ls{}, product_ms{}, product_ls{};
    std::array<wchar_t, 64> file_text{}, product_text{};
    bool fixed_valid{}, strings_complete{}, strings_conflict{};
    const char *reason{"version metadata unavailable"};
  };

  inline bool tuple(DWORD ms, DWORD ls, unsigned major, unsigned minor, unsigned patch) {
    return ms == static_cast<DWORD>(MAKELONG(minor, major)) && ls == static_cast<DWORD>(MAKELONG(0, patch));
  }
  inline bool text_matches(std::wstring_view value, abi expected) {
    if (expected == abi::v1_1_1) return value == L"1.1.1.0" || value == L"1,1,1,0";
    if (expected == abi::v2_7_30) return value == L"2.7.30.0" || value == L"2,7,30,0";
    return false;
  }
  inline abi classify(information &value) {
    if (!value.fixed_valid) return abi::unsupported;
    if (value.strings_conflict) {
      value.reason = "conflicting or malformed version strings";
      return abi::unsupported;
    }
    abi fixed = abi::unsupported;
    if (tuple(value.file_ms, value.file_ls, 1, 1, 1)) fixed = abi::v1_1_1;
    if (tuple(value.file_ms, value.file_ls, 2, 7, 30)) fixed = abi::v2_7_30;
    if (fixed != abi::unsupported) {
      if (value.file_ms != value.product_ms || value.file_ls != value.product_ls) {
        value.reason = "fixed file/product versions disagree";
        return abi::unsupported;
      }
      if ((value.file_text[0] && !text_matches(value.file_text.data(), fixed)) ||
          (value.product_text[0] && !text_matches(value.product_text.data(), fixed))) {
        value.reason = "fixed and string versions disagree";
        return abi::unsupported;
      }
      value.reason = "validated fixed version";
      return fixed;
    }
    // The pinned v1.1.1 SDK resource.h spells the numeric VERSIONINFO macro
    // as 1.1.1.0 instead of comma-separated words. Its shipping interposer has
    // fixed file/product 1.0.0.0 but exact FileVersion/ProductVersion 1.1.1.0.
    // Accept only that observed combination, not arbitrary 1.x/string versions.
    if (tuple(value.file_ms, value.file_ls, 1, 0, 0) &&
        tuple(value.product_ms, value.product_ls, 1, 0, 0) && value.strings_complete &&
        std::wstring_view(value.file_text.data()) == L"1.1.1.0" &&
        std::wstring_view(value.product_text.data()) == L"1.1.1.0") {
      value.reason = "validated 1.1.1 resource-version quirk (fixed 1.0.0.0)";
      return abi::v1_1_1;
    }
    value.reason = "unvalidated fixed/string version combination";
    return abi::unsupported;
  }

  // Version resources are read as file data. This never loads or executes a DLL.
  inline information read_file(const wchar_t *path) {
    information out;
    DWORD unused{};
    const DWORD bytes = GetFileVersionInfoSizeW(path, &unused);
    if (!bytes || bytes > 1024 * 1024) {
      out.reason = "missing or oversized version resource";
      return out;
    }
    std::vector<unsigned char> buffer;
    try { buffer.resize(bytes); }
    catch (...) { out.reason = "version resource allocation failed"; return out; }
    if (!GetFileVersionInfoW(path, 0, bytes, buffer.data())) {
      out.reason = "version resource read failed";
      return out;
    }
    VS_FIXEDFILEINFO *fixed{}; UINT length{};
    if (!VerQueryValueW(buffer.data(), L"\\", reinterpret_cast<void **>(&fixed), &length) ||
        length < sizeof(*fixed) || fixed->dwSignature != 0xfeef04bd) {
      out.reason = "invalid fixed version resource";
      return out;
    }
    out.fixed_valid = true;
    out.file_ms = fixed->dwFileVersionMS; out.file_ls = fixed->dwFileVersionLS;
    out.product_ms = fixed->dwProductVersionMS; out.product_ls = fixed->dwProductVersionLS;
    struct translation { WORD language, codepage; };
    translation *translations{};
    if (!VerQueryValueW(buffer.data(), L"\\VarFileInfo\\Translation", reinterpret_cast<void **>(&translations), &length) ||
        !length || length % sizeof(translation) || length / sizeof(translation) > 16) {
      classify(out);
      return out;
    }
    out.strings_complete = true;
    const unsigned count = length / sizeof(translation);
    for (unsigned i = 0; i < count; ++i) {
      const auto read_text = [&](const wchar_t *key, std::array<wchar_t, 64> &destination) {
        wchar_t query[96]{};
        std::swprintf(query, std::size(query), L"\\StringFileInfo\\%04x%04x\\%ls",
          translations[i].language, translations[i].codepage, key);
        wchar_t *text{}; UINT characters{};
        if (!VerQueryValueW(buffer.data(), query, reinterpret_cast<void **>(&text), &characters)) {
          out.strings_complete = false;
          return;
        }
        if (!characters || characters > destination.size() || text[characters - 1] != L'\0') {
          out.strings_conflict = true;
          return;
        }
        const std::wstring_view value(text, characters - 1);
        if (value.empty() || value.find(L'\0') != std::wstring_view::npos ||
            (destination[0] && std::wstring_view(destination.data()) != value)) {
          out.strings_conflict = true;
          return;
        }
        std::wmemcpy(destination.data(), text, characters);
      };
      read_text(L"FileVersion", out.file_text);
      read_text(L"ProductVersion", out.product_text);
    }
    classify(out);
    return out;
  }
}

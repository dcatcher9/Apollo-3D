// SPDX-License-Identifier: GPL-3.0-only
#pragma once

#include <Windows.h>
#include <array>
#include <cstdio>
#include <cwchar>
#include <string_view>
#include <vector>

namespace sunshine_streamline::versioning {
  // Public ABI families, not a list of individual game DLL builds. The early
  // 1.x resource has no state word; 2.x uses typed/versioned structures.
  enum class abi { unsupported, legacy_no_state, legacy, v2 };
  struct number {
    unsigned major{}, minor{}, patch{}, build{};
    bool operator==(const number &other) const {
      return major == other.major && minor == other.minor && patch == other.patch && build == other.build;
    }
  };
  struct information {
    DWORD file_ms{}, file_ls{}, product_ms{}, product_ls{};
    std::array<wchar_t, 64> file_text{}, product_text{};
    bool fixed_valid{}, strings_complete{}, strings_conflict{};
    const char *reason{"version metadata unavailable"};
  };

  inline bool parse(std::wstring_view text, number &out) {
    unsigned parts[4]{};
    std::size_t position{};
    wchar_t separator{};
    for (unsigned i = 0; i != 4; ++i) {
      while (position < text.size() && text[position] == L' ') ++position;
      const auto start = position;
      while (position < text.size() && text[position] >= L'0' && text[position] <= L'9') {
        parts[i] = parts[i] * 10 + unsigned(text[position++] - L'0');
        if (parts[i] > 65535) return false;
      }
      if (position == start) return false;
      while (position < text.size() && text[position] == L' ') ++position;
      if (i == 3) {
        if (position != text.size()) return false;
      } else {
        if (position == text.size() || (text[position] != L'.' && text[position] != L',')) return false;
        if (separator && text[position] != separator) return false;
        separator = text[position++];
      }
    }
    out = {parts[0], parts[1], parts[2], parts[3]};
    return true;
  }
  inline bool identity(information &value, number &out) {
    if (!value.fixed_valid) return false;
    if (value.strings_conflict) {
      value.reason = "conflicting or malformed version strings";
      return false;
    }
    if (value.file_ms != value.product_ms || value.file_ls != value.product_ls) {
      value.reason = "fixed file/product versions disagree";
      return false;
    }
    out = {HIWORD(value.file_ms), LOWORD(value.file_ms), HIWORD(value.file_ls), LOWORD(value.file_ls)};
    number file{}, product{};
    const bool has_file = value.file_text[0] != 0, has_product = value.product_text[0] != 0;
    if ((has_file && !parse(value.file_text.data(), file)) ||
        (has_product && !parse(value.product_text.data(), product)) ||
        (has_file && has_product && !(file == product))) {
      value.reason = "conflicting or malformed version strings";
      return false;
    }
    // Released 1.x resource scripts used a dotted numeric VERSIONINFO macro,
    // producing fixed 1.0.0.0. Only the surveyed release range may use this
    // exception, with complete, agreeing strings. It never overrides a 2.x ABI.
    if (out == number{1, 0, 0, 0} && value.strings_complete && has_file && has_product &&
        file.major == 1 && file.build == 0 &&
        ((file.minor == 0 && file.patch <= 4) || (file.minor == 1 && file.patch <= 1))) {
      out = file;
      value.reason = "validated 1.x resource-version quirk (fixed 1.0.0.0)";
      return true;
    }
    // These two official releases have the same dotted-resource defect. Keep
    // it bounded to their observed identity pairs, independently of ABI choice.
    if (out == number{2, 0, 0, 0} && value.strings_complete && has_file && has_product &&
        (file == number{2, 0, 1, 0} || file == number{2, 1, 0, 0})) {
      out = file;
      value.reason = "validated early 2.x resource-version quirk (fixed 2.0.0.0)";
      return true;
    }
    if ((has_file && !(file == out)) || (has_product && !(product == out))) {
      value.reason = "fixed and string versions disagree";
      return false;
    }
    value.reason = "consistent version identity";
    return true;
  }
  inline abi classify(information &value) {
    number version;
    if (!identity(value, version)) return abi::unsupported;
    if (version.major == 2) return abi::v2;
    if (version.major == 1) return version.minor == 0 && version.patch < 3 ? abi::legacy_no_state : abi::legacy;
    value.reason = "unsupported Streamline ABI major";
    return abi::unsupported;
  }
  inline bool private_state_v1_1_1(information value) {
    number version;
    return identity(value, version) && version == number{1, 1, 1, 0};
  }
  inline bool is_legacy(abi value) { return value == abi::legacy_no_state || value == abi::legacy; }
  inline const char *name(abi value) {
    switch (value) {
    case abi::legacy_no_state: return "SL 1.0.0-1.0.2 ABI (resource without state)";
    case abi::legacy: return "SL 1.x ABI (resource with state)";
    case abi::v2: return "SL 2.x ABI (versioned structures)";
    default: return "unsupported observer ABI";
    }
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

// SPDX-License-Identifier: GPL-3.0-only
// The synthetic Streamline interposers themselves (CMakeLists.txt, interposer
// fixtures), checked without a device before a runtime fixture relies on one:
// each DLL's version resource names its configured release, it exports
// slSetTagForFrame exactly where configured, its options function accepts
// what that release's header declares and refuses the rest, and
// SunshineFixtureSetResults sets the result of every boundary.
//
// Usage: reshade_sl_fixture_interposer_tests (<dll> <a,b,c,d> <FRAME_TAG|NO_FRAME_TAG>)...
#include "streamline_camera_data.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>

namespace {
  using namespace sunshine_streamline;

  void require(bool condition, const std::string &message) {
    if (!condition) {
      throw std::runtime_error(message);
    }
  }

  struct version {
    unsigned part[4] {};
  };

  bool parse(const char *text, version &out) {
    for (unsigned i = 0; i != 4; ++i) {
      char *end {};
      out.part[i] = unsigned(std::strtoul(text, &end, 10));
      if (end == text || (i != 3 && *end != ',' && *end != '.') || (i == 3 && *end)) {
        return false;
      }
      text = end + (i != 3);
    }
    return true;
  }

  bool operator==(const version &a, const version &b) {
    return !std::memcmp(a.part, b.part, sizeof(a.part));
  }

  // What each configured release's own sl_dlss_g.h declares (survey headers):
  // DLSSGMode values before eCount (Off, On and Auto from 2.2.1, Dynamic from
  // 2.11.1), and the largest numFramesToGenerate (1 before multi frame
  // generation, documented from 2.7.2; the fixture's device reports 5, the
  // largest any surveyed header documents). 2.6.10 lies between the surveyed
  // 2.4.15 and 2.7.2 and declares what 2.4.15 does; 2.15.0 and 3.0.0, newer
  // than the survey, declare what 2.14.1 does.
  struct declaration {
    version release;
    std::uint32_t modes, frames;
  };

  const declaration declarations[] {
    {{{2, 6, 10, 0}}, 3, 1},
    {{{2, 7, 30, 0}}, 3, 5},
    {{{2, 14, 1, 0}}, 4, 5},
    {{{2, 15, 0, 0}}, 4, 5},
    {{{3, 0, 0, 0}}, 4, 5},
  };

  constexpr guid options_type {0xfac5f1cb, 0x2dfd, 0x4f36, {0xa1, 0xe6, 0x3a, 0x9e, 0x86, 0x52, 0x56, 0xc5}};
  constexpr std::int32_t refused = 1;

  version file_version(const wchar_t *path, std::string &text, std::string &product_text) {
    DWORD unused {};
    const auto bytes = GetFileVersionInfoSizeW(path, &unused);
    require(bytes, "The interposer has no version resource");
    std::vector<unsigned char> buffer(bytes);
    require(GetFileVersionInfoW(path, 0, bytes, buffer.data()), "The interposer's version resource is unreadable");
    VS_FIXEDFILEINFO *fixed {};
    UINT length {};
    require(VerQueryValueW(buffer.data(), L"\\", reinterpret_cast<void **>(&fixed), &length) && length >= sizeof(*fixed) && fixed->dwSignature == 0xfeef04bd && fixed->dwFileVersionMS == fixed->dwProductVersionMS && fixed->dwFileVersionLS == fixed->dwProductVersionLS, "The interposer's fixed file and product versions differ");
    const auto read_text = [&](const wchar_t *key) {
      wchar_t query[64] {};
      std::swprintf(query, 64, L"\\StringFileInfo\\040904b0\\%ls", key);
      wchar_t *value {};
      UINT characters {};
      require(VerQueryValueW(buffer.data(), query, reinterpret_cast<void **>(&value), &characters) && characters, "The interposer's version strings are missing");
      std::string out;
      for (UINT i = 0; i != characters && value[i]; ++i) {
        out += char(value[i]);
      }
      return out;
    };
    text = read_text(L"FileVersion");
    product_text = read_text(L"ProductVersion");
    return {{HIWORD(fixed->dwFileVersionMS), LOWORD(fixed->dwFileVersionMS), HIWORD(fixed->dwFileVersionLS), LOWORD(fixed->dwFileVersionLS)}};
  }

  template<class T>
  T entry(HMODULE module, const char *name) {
    return reinterpret_cast<T>(GetProcAddress(module, name));
  }

  void check(const char *path, const char *configured, bool frame_tags) {
    const std::string at = std::string(path) + ": ";
    version expected;
    require(parse(configured, expected), at + "unreadable configured version");
    const declaration *declared {};
    for (const auto &value : declarations) {
      if (value.release == expected) {
        declared = &value;
      }
    }
    require(declared, at + "no declaration for this release: add its header's DLSSGMode values and frame count");
    std::wstring wide(path, path + std::strlen(path));
    std::string text, product_text;
    version named;
    require(file_version(wide.c_str(), text, product_text) == expected, at + "the fixed version is not the configured release");
    require(parse(text.c_str(), named) && named == expected && product_text == text, at + "the version strings do not name the configured release: " + text);

    const auto module = LoadLibraryW(wide.c_str());
    require(module, at + "cannot load");

    struct unload {
      HMODULE module;

      ~unload() {
        FreeLibrary(module);
      }
    } scope {module};

    const auto constants = entry<abi_v2::set_constants>(module, "slSetConstants");
    const auto tag = entry<abi_v2::set_tag>(module, "slSetTag");
    const auto frame_tag = entry<abi_v2::set_tag_for_frame>(module, "slSetTagForFrame");
    const auto new_token = entry<abi_v2::get_new_frame_token>(module, "slGetNewFrameToken");
    const auto evaluate = entry<abi_v2::evaluate_feature>(module, "slEvaluateFeature");
    const auto feature_function = entry<abi_v2::get_feature_function>(module, "slGetFeatureFunction");
    const auto set_results = entry<void (*)(std::int32_t, std::int32_t, std::int32_t, std::int32_t, std::int32_t)>(module, "SunshineFixtureSetResults");
    const auto set_options_result = entry<void (*)(std::int32_t)>(module, "SunshineFixtureSetOptionsResult");
    require(constants && tag && new_token && evaluate && feature_function && set_results && set_options_result, at + "an SDK or fixture export is missing");
    require(bool(frame_tag) == frame_tags, at + (frame_tags ? "slSetTagForFrame is missing" : "slSetTagForFrame is exported before 2.7"));
    void *function {};
    require(feature_function(1000, "slDLSSGGetState", function) != 0 && !function, at + "an unknown feature function was handed out");
    require(feature_function(1000, "slDLSSGSetOptions", function) == 0 && function, at + "the options function is missing");
    const auto set_options = reinterpret_cast<abi_v2::fg_set_options>(function);

    // Every mode 0-4 with 0-6 frames, at struct versions 1, 3, 5 and 6 (a
    // version above any surveyed one is read by its prefix).
    const abi_v2::viewport viewport {{nullptr, viewport_guid, 1}, 0};
    unsigned accepted {};
    for (std::uint32_t struct_version : {1u, 3u, 5u, 6u}) {
      for (std::uint32_t mode = 0; mode != 5; ++mode) {
        for (std::uint32_t frames = 0; frames != 7; ++frames) {
          const abi_v2::fg_options_prefix options {{nullptr, options_type, struct_version}, mode, frames};
          const bool legal = mode < declared->modes && (!mode || (frames && frames <= declared->frames));
          accepted += legal;
          require(set_options(viewport, options) == (legal ? 0 : refused), at + "options mode " + std::to_string(mode) + " frames " + std::to_string(frames) + " version " + std::to_string(struct_version) + (legal ? " were refused" : " were accepted"));
        }
      }
    }
    abi_v2::fg_options_prefix unversioned {{nullptr, options_type, 0}, 1, 1}, foreign {{nullptr, viewport_guid, 1}, 1, 1};
    require(set_options(viewport, unversioned) == refused && set_options(viewport, foreign) == refused, at + "options of version 0 or another type were accepted");

    // Each boundary returns its own configured result for a call it accepts.
    abi_v2::frame_token *token {};
    require(new_token(token, nullptr) == 0 && token, at + "no frame token");
    const abi_v2::constants camera {};
    const abi_v2::fg_options_prefix on {{nullptr, options_type, 3}, 1, 1}, undeclared {{nullptr, options_type, 3}, 4, 1};
    set_results(11, 12, 13, 14, 15);
    function = nullptr;
    require(constants(camera, *token, viewport) == 11 && tag(viewport, nullptr, 0, nullptr) == 12 && (!frame_tag || frame_tag(*token, viewport, nullptr, 0, nullptr) == 12) && evaluate(0, *token, nullptr, 0, nullptr) == 13 && set_options(viewport, on) == 14 && set_options(viewport, undeclared) == refused && feature_function(1000, "slDLSSGSetOptions", function) == 15 && function, at + "a boundary did not return its configured result");
    set_options_result(39);
    require(set_options(viewport, on) == 39 && tag(viewport, nullptr, 0, nullptr) == 12, at + "the options result alone was not set");
    set_results(0, 0, 0, 0, 0);
    require(constants(camera, *token, viewport) == 0 && set_options(viewport, on) == 0 && feature_function(1000, "slDLSSGSetOptions", function) == 0, at + "results did not return to eOk");
    std::printf("PASS %s: release %s, frame tags=%u, %u of 140 option sets accepted, settable results\n", path, configured, unsigned(frame_tags), accepted);
  }
}  // namespace

int main(int argc, char **argv) {
  if (argc < 4 || (argc - 1) % 3) {
    std::fputs("usage: reshade_sl_fixture_interposer_tests (<dll> <a,b,c,d> <FRAME_TAG|NO_FRAME_TAG>)...\n", stderr);
    return 2;
  }
  try {
    for (int i = 1; i < argc; i += 3) {
      require(!std::strcmp(argv[i + 2], "FRAME_TAG") || !std::strcmp(argv[i + 2], "NO_FRAME_TAG"), std::string("unknown frame tag setting ") + argv[i + 2]);
      check(argv[i], argv[i + 1], !std::strcmp(argv[i + 2], "FRAME_TAG"));
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}

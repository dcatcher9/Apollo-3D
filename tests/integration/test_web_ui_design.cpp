/**
 * @file tests/integration/test_web_ui_design.cpp
 * @brief Guards the shared Moonlight 3D/Sunshine 3D UI design contract against drift.
 */
#include "../tests_common.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <ranges>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {
  namespace fs = std::filesystem;

  const fs::path web_root =
    fs::path(SUNSHINE_SOURCE_DIR) / "src_assets/common/assets/web";

  std::string read_source(const fs::path &path) {
    std::ifstream stream(path, std::ios::binary);
    EXPECT_TRUE(stream.is_open()) << "Unable to read " << path;
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  }

  std::vector<fs::path> styled_sources() {
    std::vector<fs::path> result;
    for (const auto &entry : fs::recursive_directory_iterator(web_root)) {
      if (!entry.is_regular_file()) {
        continue;
      }

      const auto extension = entry.path().extension();
      if (extension == ".html" || extension == ".vue" || extension == ".css") {
        result.push_back(entry.path());
      }
    }
    return result;
  }

  std::vector<std::string> style_blocks(const fs::path &path) {
    const auto source = read_source(path);
    if (path.extension() == ".css") {
      return {source};
    }

    std::vector<std::string> blocks;
    static const std::regex style_pattern(
      R"(<style(?:\s[^>]*)?>([\s\S]*?)</style>)",
      std::regex::icase
    );
    for (std::sregex_iterator it(source.begin(), source.end(), style_pattern), end;
         it != end;
         ++it) {
      blocks.push_back((*it)[1].str());
    }
    return blocks;
  }

  std::string lowercase(std::string value) {
    std::ranges::transform(value, value.begin(), [](const unsigned char character) {
      return static_cast<char>(std::tolower(character));
    });
    return value;
  }
}  // namespace

TEST(WebUiDesign, DeclaresSharedPaletteAndScales) {
  const auto css = read_source(web_root / "public/assets/css/apollo.css");

  const std::vector<std::string_view> required_contract {
    "--apollo-surface: #16181e",
    "--apollo-surface-raised: #22262b",
    "--apollo-surface-sunken: #0c0f14",
    "--apollo-border: #3c4043",
    "--apollo-accent: #8ab4f8",
    "--apollo-accent-bright: #d7e5ff",
    "--apollo-accent-deep: #36587f",
    "--apollo-text-primary: #ffffff",
    "--apollo-text-secondary: #b0b9c6",
    "--apollo-text-disabled: #71808f",
    "--apollo-status-ok: #5cd65c",
    "--apollo-status-warn: #e0b020",
    "--apollo-danger: #ffb4ab",
    "--apollo-danger-container: #6d3a3e",
    "--apollo-text-caption: 0.875rem",
    "--apollo-text-body: 1rem",
    "--apollo-text-emphasis: 1.1875rem",
    "--apollo-text-title: 1.4375rem",
    "--apollo-text-display: 1.75rem",
    "--apollo-radius-control: 8px",
    "--apollo-radius-card: 14px",
    "--apollo-radius-pill: 24px",
    "--apollo-space-xs: 4px",
    "--apollo-space-sm: 8px",
    "--apollo-space-md: 12px",
    "--apollo-space-lg: 18px",
    "--apollo-space-xl: 24px",
    "--apollo-control-compact: 40px",
    "--apollo-control-standard: 56px",
    "--apollo-control-primary: 64px",
    "--apollo-icon-inline: 32px",
    "--apollo-icon-tile: 48px",
    "--apollo-active: var(--apollo-accent)",
  };

  for (const auto declaration : required_contract) {
    EXPECT_NE(css.find(declaration), std::string::npos)
      << "Missing shared design declaration: " << declaration;
  }
}

TEST(WebUiDesign, EveryApolloVariableUseHasADeclaration) {
  static const std::regex definition_pattern(R"((--apollo-[a-z0-9-]+)\s*:)");
  static const std::regex use_pattern(R"(var\((--apollo-[a-z0-9-]+))");
  std::set<std::string> definitions;
  std::map<std::string, std::set<std::string>> uses;

  for (const auto &path : styled_sources()) {
    const auto source = read_source(path);
    for (std::sregex_iterator it(source.begin(), source.end(), definition_pattern), end;
         it != end;
         ++it) {
      definitions.insert((*it)[1].str());
    }
    for (std::sregex_iterator it(source.begin(), source.end(), use_pattern), end;
         it != end;
         ++it) {
      uses[(*it)[1].str()].insert(fs::relative(path, web_root).generic_string());
    }
  }

  for (const auto &[token, files] : uses) {
    EXPECT_TRUE(definitions.contains(token))
      << token << " is used but never declared ("
      << [&files] {
           std::ostringstream joined;
           for (const auto &file : files) {
             if (joined.tellp() > 0) {
               joined << ", ";
             }
             joined << file;
           }
           return joined.str();
         }()
      << ')';
  }
}

TEST(WebUiDesign, PageChromeUsesSemanticColorTypeAndRadiusRoles) {
  const std::map<std::string, std::set<std::string>> allowed_artwork_colors {
    {"apps.html", {"#111", "#fff"}},
    {"pin.html", {"#fff"}},  // QR code quiet zone.
    {"troubleshooting.html", {"#10141b", "#d9e1ee"}},  // Log terminal.
  };
  static const std::regex color_pattern(R"(#[0-9a-fA-F]{3,8}\b)");
  static const std::regex type_pattern(R"(font-size\s*:\s*([^;]+);)");
  static const std::regex radius_pattern(R"(border-radius\s*:\s*([^;]+);)");

  for (const auto &path : styled_sources()) {
    if (path.extension() == ".css") {
      continue;  // The central token layer owns the literal declarations.
    }

    const auto relative = fs::relative(path, web_root).generic_string();
    for (const auto &style : style_blocks(path)) {
      for (std::sregex_iterator it(style.begin(), style.end(), color_pattern), end;
           it != end;
           ++it) {
        const auto color = lowercase((*it)[0].str());
        const auto allowed = allowed_artwork_colors.find(relative);
        EXPECT_TRUE(
          allowed != allowed_artwork_colors.end() && allowed->second.contains(color)
        ) << relative << " introduces a raw UI color " << color;
      }

      for (std::sregex_iterator it(style.begin(), style.end(), type_pattern), end;
           it != end;
           ++it) {
        const auto value = (*it)[1].str();
        const bool artwork_scale =
          relative == "apps.html" && value.starts_with("clamp(");
        EXPECT_TRUE(value.find("var(--apollo-text-") != std::string::npos || artwork_scale)
          << relative << " introduces a raw type step: " << value;
      }

      for (std::sregex_iterator it(style.begin(), style.end(), radius_pattern), end;
           it != end;
           ++it) {
        const auto value = (*it)[1].str();
        EXPECT_TRUE(
          value.find("var(--apollo-radius-") != std::string::npos || value == "50%"
        ) << relative << " introduces a raw radius: " << value;
      }
    }
  }
}

TEST(WebUiDesign, PairingKeepsPinPrimaryAndQrSecondary) {
  const auto source = read_source(web_root / "pin.html");
  const auto pin = source.find("id=\"PIN\"");
  const auto qr = source.find("Scan QR code");

  ASSERT_NE(pin, std::string::npos);
  ASSERT_NE(qr, std::string::npos);
  EXPECT_LT(pin, qr);
  EXPECT_NE(source.find("@submit.prevent=\"pairWithPin\""), std::string::npos);
  EXPECT_EQ(source.find("Pair another way"), std::string::npos);
}

TEST(WebUiDesign, AuthenticationEntriesUseSharedLayouts) {
  const auto login = read_source(web_root / "login.html");
  const auto welcome = read_source(web_root / "welcome.html");
  const auto password = read_source(web_root / "password.html");

  EXPECT_NE(login.find("apollo-auth-shell is-compact"), std::string::npos);
  EXPECT_NE(welcome.find("apollo-auth-shell"), std::string::npos);
  EXPECT_NE(password.find("apollo-password-grid"), std::string::npos);
}

TEST(WebUiDesign, PresentsSunshineAndMoonlightBrandPair) {
  const auto navbar = read_source(web_root / "Navbar.vue");
  const auto header = read_source(web_root / "template_header.html");
  const auto pairing = read_source(web_root / "pin.html");

  EXPECT_NE(navbar.find("<strong>Sunshine 3D</strong>"), std::string::npos);
  EXPECT_NE(navbar.find("logo-sunshine3d-45.png"), std::string::npos);
  EXPECT_NE(header.find("<title>Sunshine 3D</title>"), std::string::npos);
  EXPECT_NE(header.find("images/sunshine3d.ico"), std::string::npos);
  EXPECT_NE(pairing.find("Pair Moonlight 3D"), std::string::npos);

  EXPECT_TRUE(fs::exists(web_root / "public/images/logo-sunshine3d-45.png"));
  EXPECT_TRUE(fs::exists(web_root / "public/images/sunshine3d.ico"));
  EXPECT_EQ(navbar.find("Apollo XR"), std::string::npos);
  EXPECT_EQ(pairing.find("Pair Artemis"), std::string::npos);
}

TEST(WebUiDesign, LocaleValuesDoNotExposeRetiredProductNames) {
  const std::regex retired_value {
    R"(:\s*"[^"\r\n]*(Apollo|Artemis)[^"\r\n]*")",
    std::regex::icase
  };
  const auto locale_root = web_root / "public/assets/locale";
  for (const auto &entry : fs::directory_iterator(locale_root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json") {
      continue;
    }
    const auto locale = read_source(entry.path());
    EXPECT_FALSE(std::regex_search(locale, retired_value))
      << "Retired branding remains in a visible value in "
      << entry.path().filename();
  }
}

TEST(WebUiDesign, KeepsTaskbarRepairDirectlyUnderTheTraySetting) {
  const auto essentials =
    read_source(web_root / "configs/tabs/Essentials.vue");
  const auto tray_section = essentials.find(
    "<section class=\"settings-card settings-card-wide tray-setting\">"
  );
  const auto keep_tray =
    essentials.find("Keep Sunshine 3D in the system tray", tray_section);
  const auto repair =
    essentials.find("config.virtual_display_restart_explorer", keep_tray);
  const auto tray_section_end = essentials.find("</section>", tray_section);

  ASSERT_NE(tray_section, std::string::npos);
  ASSERT_NE(keep_tray, std::string::npos);
  ASSERT_NE(repair, std::string::npos);
  ASSERT_NE(tray_section_end, std::string::npos);
  EXPECT_LT(tray_section, keep_tray);
  EXPECT_LT(keep_tray, repair);
  EXPECT_LT(repair, tray_section_end);

  const auto virtual_display = essentials.find("<h2>Virtual display</h2>");
  const auto virtual_display_end =
    essentials.find("</section>", virtual_display);
  ASSERT_NE(virtual_display, std::string::npos);
  ASSERT_NE(virtual_display_end, std::string::npos);
  EXPECT_EQ(
    essentials.substr(
      virtual_display,
      virtual_display_end - virtual_display
    ).find("config.virtual_display_restart_explorer"),
    std::string::npos
  );
}

TEST(WebUiDesign, TaskbarRepairUiMatchesItsOptInNativeDefault) {
  const auto schema = read_source(web_root / "config.html");
  const auto locale =
    read_source(web_root / "public/assets/locale/en.json");
  const auto documentation =
    read_source(fs::path(SUNSHINE_SOURCE_DIR) / "docs/configuration.md");

  EXPECT_NE(
    schema.find("\"virtual_display_restart_explorer\": \"off\""),
    std::string::npos
  );
  EXPECT_NE(locale.find("\"Off by default."), std::string::npos);
  const auto setting = documentation.find("### virtual_display_restart_explorer");
  ASSERT_NE(setting, std::string::npos);
  const auto next_setting = documentation.find("\n## ", setting);
  ASSERT_NE(next_setting, std::string::npos);
  const auto section = documentation.substr(setting, next_setting - setting);
  EXPECT_NE(section.find("<td>Default</td>"), std::string::npos);
  EXPECT_NE(section.find("\n            off\n"), std::string::npos);
}

TEST(WebUiDesign, HostSbsSceneControllerExposesOnlyImplementedBackends) {
  const auto advanced =
    read_source(web_root / "configs/tabs/Advanced.vue");
  const auto schema = read_source(web_root / "config.html");
  const auto locale =
    read_source(web_root / "public/assets/locale/en.json");
  const auto documentation =
    read_source(fs::path(SUNSHINE_SOURCE_DIR) / "docs/configuration.md");

  const auto select_begin = advanced.find(
    "<select id=\"sbs_scene_controller\""
  );
  ASSERT_NE(select_begin, std::string::npos);
  const auto select_end = advanced.find("</select>", select_begin);
  ASSERT_NE(select_end, std::string::npos);
  const auto select =
    advanced.substr(select_begin, select_end - select_begin);

  static const std::regex option_pattern(
    R"OPTION(<option value="([^"]+)")OPTION"
  );
  std::vector<std::string> values;
  for (std::sregex_iterator it(select.begin(), select.end(), option_pattern), end;
       it != end;
       ++it) {
    values.push_back((*it)[1].str());
  }
  EXPECT_EQ(
    values,
    (std::vector<std::string> {"off", "shadow_rules"})
  );

  EXPECT_NE(
    schema.find("\"sbs_scene_controller\": \"off\""),
    std::string::npos
  );
  EXPECT_NE(
    locale.find("Shadow rules records diagnostics but never changes rendered output."),
    std::string::npos
  );
  EXPECT_EQ(locale.find("\"sbs_scene_controller_rules\""), std::string::npos);

  const auto setting = documentation.find("### sbs_scene_controller");
  ASSERT_NE(setting, std::string::npos);
  const auto next_setting = documentation.find("\n## ", setting);
  ASSERT_NE(next_setting, std::string::npos);
  const auto section = documentation.substr(setting, next_setting - setting);
  EXPECT_NE(section.find("<code>shadow_rules</code>"), std::string::npos);
  EXPECT_NE(section.find("<code>off</code>"), std::string::npos);
}

TEST(WebUiDesign, OfflineConversionKeepsNativeJobApiIsolatedAndAuditable) {
  const auto page = read_source(web_root / "offline-conversion.html");
  const auto api = read_source(web_root / "offline-sbs-api.js");
  const auto navbar = read_source(web_root / "Navbar.vue");
  const auto vite = read_source(fs::path(SUNSHINE_SOURCE_DIR) / "vite.config.js");
  const auto http = read_source(fs::path(SUNSHINE_SOURCE_DIR) / "src/confighttp.cpp");
  const auto jobs =
    read_source(fs::path(SUNSHINE_SOURCE_DIR) / "src/offline_sbs_job.cpp");

  EXPECT_NE(page.find("v-model.trim=\"form.inputPath\""), std::string::npos);
  EXPECT_NE(page.find("v-model.trim=\"form.outputName\""), std::string::npos);
  EXPECT_NE(page.find("v-model=\"form.operation\""), std::string::npos);
  EXPECT_NE(page.find("value=\"convert\""), std::string::npos);
  EXPECT_NE(page.find("value=\"evaluate\""), std::string::npos);
  EXPECT_NE(page.find("Analyze only"), std::string::npos);
  EXPECT_NE(
    page.find("if (this.form.operation === 'convert')"),
    std::string::npos
  );
  EXPECT_NE(page.find("request.output_name ="), std::string::npos);
  EXPECT_NE(page.find("const operation = settings.operation"), std::string::npos);
  EXPECT_NE(page.find("one non-empty regular video file"), std::string::npos);
  EXPECT_EQ(page.find("numbered frame directory"), std::string::npos);
  EXPECT_NE(page.find("max=\"64\""), std::string::npos);
  EXPECT_NE(page.find("budget <= 64"), std::string::npos);
  EXPECT_NE(page.find("value=\"hevc_nvenc\""), std::string::npos);
  EXPECT_NE(page.find("value=\"av1_nvenc\""), std::string::npos);
  EXPECT_NE(page.find("<strong>H.265 / HEVC</strong>"), std::string::npos);
  EXPECT_NE(page.find("<strong>AV1</strong>"), std::string::npos);
  EXPECT_EQ(page.find("<strong>10-bit H.265 / HEVC</strong>"), std::string::npos);
  EXPECT_EQ(page.find("<strong>10-bit AV1</strong>"), std::string::npos);
  EXPECT_NE(
    page.find("Static HDR uses 10-bit output; SDR uses 8-bit output."),
    std::string::npos
  );
  EXPECT_NE(page.find("scene_cache_max_bytes"), std::string::npos);
  EXPECT_NE(page.find("cache_budget_policy"), std::string::npos);
  EXPECT_NE(
    page.find("v-if=\"form.operation === 'convert'\" class=\"offline-form-row\""),
    std::string::npos
  );
  EXPECT_NE(
    page.find("this.form.operation === 'evaluate' ||"),
    std::string::npos
  );
  EXPECT_NE(page.find("value=\"fail\""), std::string::npos);
  EXPECT_NE(page.find("value=\"split\""), std::string::npos);
  EXPECT_NE(page.find("Recent scene decisions"), std::string::npos);
  EXPECT_NE(page.find("Download full audit"), std::string::npos);
  EXPECT_NE(page.find("sceneTotal"), std::string::npos);
  EXPECT_NE(page.find("latest {{ sceneDecisions.length }} shown"), std::string::npos);
  EXPECT_NE(page.find("@click=\"cancelJob(selectedJob)\""), std::string::npos);
  EXPECT_NE(page.find("Interrupted — not resumable in version 1"), std::string::npos);
  EXPECT_NE(page.find("@click=\"restartJob(selectedJob)\""), std::string::npos);
  EXPECT_NE(page.find("Windows and NVIDIA only"), std::string::npos);
  EXPECT_NE(page.find("Static HDR is supported"), std::string::npos);
  EXPECT_NE(page.find("Dolby Vision and dynamic"), std::string::npos);
  EXPECT_NE(page.find("'publishing'"), std::string::npos);
  EXPECT_NE(page.find("stream before starting a job"), std::string::npos);
  EXPECT_NE(page.find("built-in job manager"), std::string::npos);
  EXPECT_EQ(page.find("native service"), std::string::npos);
  EXPECT_EQ(api.find("job service"), std::string::npos);

  // HTTP details belong to one browser boundary, not the page component.
  EXPECT_EQ(page.find("fetch("), std::string::npos);
  EXPECT_NE(api.find("OFFLINE_SBS_JOBS_URL = '/api/offline-sbs/jobs'"), std::string::npos);
  EXPECT_NE(api.find("credentials: 'include'"), std::string::npos);
  EXPECT_NE(api.find("createOfflineSbsJob"), std::string::npos);
  EXPECT_NE(api.find("cancelOfflineSbsJob"), std::string::npos);
  EXPECT_NE(api.find("downloadOfflineSbsSceneAudit"), std::string::npos);
  EXPECT_NE(api.find("/scene-audit"), std::string::npos);
  EXPECT_NE(api.find("getOfflineSbsOverview"), std::string::npos);
  EXPECT_NE(api.find("getOfflineSbsJob"), std::string::npos);
  EXPECT_NE(
    api.find("`${OFFLINE_SBS_JOBS_URL}/${requireJobId(jobId)}`"),
    std::string::npos
  );
  EXPECT_NE(page.find("selectedJobDetail"), std::string::npos);
  EXPECT_NE(page.find("loadSelectedJob"), std::string::npos);
  EXPECT_NE(page.find("detailRequestSerial"), std::string::npos);
  EXPECT_NE(page.find("overviewAbortController?.abort()"), std::string::npos);
  EXPECT_NE(
    page.find("if (silent && this.refreshing) return"),
    std::string::npos
  );
  EXPECT_NE(page.find("capabilities?.codecs"), std::string::npos);
  EXPECT_NE(page.find("live_stream_active"), std::string::npos);
  EXPECT_NE(page.find(":disabled=\"!codecAvailable('hevc_nvenc')\""), std::string::npos);
  EXPECT_NE(page.find("End the current stream before starting"), std::string::npos);
  EXPECT_NE(page.find("Live updates are delayed"), std::string::npos);
  EXPECT_NE(page.find("this.pollWarning = message"), std::string::npos);
  EXPECT_NE(page.find("retrying automatically"), std::string::npos);
  EXPECT_NE(
    page.find(":aria-label=\"jobProgressLabel(selectedJob)\""),
    std::string::npos
  );
  EXPECT_NE(page.find("Offline analysis progress"), std::string::npos);
  EXPECT_NE(
    page.find("aria-describedby=\"offlineInputPathHelp\""),
    std::string::npos
  );
  EXPECT_NE(
    page.find("aria-describedby=\"offlineOutputNameHelp\""),
    std::string::npos
  );
  EXPECT_NE(
    page.find("aria-describedby=\"offlineCacheBudgetHelp\""),
    std::string::npos
  );
  EXPECT_NE(
    page.find("aria-describedby=\"offlineBudgetPolicyHelp\""),
    std::string::npos
  );
  EXPECT_EQ(api.find("/resume"), std::string::npos);
  EXPECT_NE(
    http.find("/scene-audit$"),
    std::string::npos
  );
  EXPECT_NE(http.find("getOfflineSbsSceneAudit"), std::string::npos);
  const auto audit_handler = http.find("void getOfflineSbsSceneAudit");
  const auto cancel_handler = http.find("void cancelOfflineSbsJob", audit_handler);
  ASSERT_NE(audit_handler, std::string::npos);
  ASSERT_NE(cancel_handler, std::string::npos);
  const auto audit_handler_source =
    http.substr(audit_handler, cancel_handler - audit_handler);
  EXPECT_NE(
    audit_handler_source.find("if (!authenticate(response, request))"),
    std::string::npos
  );
  EXPECT_NE(
    jobs.find("max_scene_audit_bytes = 32ull * 1024ull * 1024ull"),
    std::string::npos
  );
  EXPECT_NE(
    jobs.find("record->worker.result_directory / \"scene-audit.json\""),
    std::string::npos
  );
  EXPECT_NE(
    jobs.find(
      "summary.progress.scene_decisions = nlohmann::json::array()"
    ),
    std::string::npos
  );
  EXPECT_NE(
    jobs.find("contract size changed while it was being read"),
    std::string::npos
  );

  EXPECT_NE(navbar.find("href: './offline-conversion'"), std::string::npos);
  EXPECT_NE(
    vite.find("offlineConversion: resolve(assetsSrcPath, 'offline-conversion.html')"),
    std::string::npos
  );

  // The page and APIs follow the normal Web UI authentication policy, including the trusted-local
  // behavior. The POST body has both an endpoint-specific limit and a bounded server stream buffer
  // for requests without a trustworthy Content-Length.
  EXPECT_NE(
    http.find("if (!authenticate(response, request, true))"),
    std::string::npos
  );
  EXPECT_EQ(http.find("authenticateOfflineSbs"), std::string::npos);
  EXPECT_NE(
    http.find("!validateOfflineSbsRequestSize(response, request)"),
    std::string::npos
  );
  EXPECT_NE(
    http.find(
      "server.config.max_request_streambuf_size = WEB_UI_REQUEST_STREAM_MAX_BYTES"
    ),
    std::string::npos
  );
  EXPECT_NE(
    http.find("including on a keep-alive connection"),
    std::string::npos
  );
  EXPECT_NE(
    http.find("server.config.request_content_length_limit"),
    std::string::npos
  );
}

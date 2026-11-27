/**
 * @file src/confighttp_routes.h
 * @brief The Web UI route handlers, by the file of their domain. confighttp.cpp registers them.
 */
#pragma once

// lib includes
#include <nlohmann/json_fwd.hpp>

// local includes
#include "confighttp_common.h"

namespace platf::game3d {
  struct setup_status_t;
}  // namespace platf::game3d

namespace confighttp {
  // Pages and static assets: confighttp_pages.cpp
  void getIndexPage(resp_https_t response, req_https_t request);
  void getPinPage(resp_https_t response, req_https_t request);
  void getAppsPage(resp_https_t response, req_https_t request);
  void getConfigPage(resp_https_t response, req_https_t request);
  void getLoginPage(resp_https_t response, req_https_t request);
  void getWelcomePage(resp_https_t response, req_https_t request);
  void getTroubleshootingPage(resp_https_t response, req_https_t request);
  void getOfflineConversionPage(resp_https_t response, req_https_t request);
  void getFaviconImage(resp_https_t response, req_https_t request);
  void getProductLogoImage(resp_https_t response, req_https_t request);
  void getNodeModules(resp_https_t response, req_https_t request);

  // The session: confighttp_common.cpp
  void login(resp_https_t response, req_https_t request);
  void savePassword(resp_https_t response, req_https_t request);

  // The app library and covers: confighttp_apps.cpp
  void getApps(resp_https_t response, req_https_t request);
  void saveApp(resp_https_t response, req_https_t request);
  void closeApp(resp_https_t response, req_https_t request);
  void reorderApps(resp_https_t response, req_https_t request);
  void deleteApp(resp_https_t response, req_https_t request);
  void uploadCover(resp_https_t response, req_https_t request);

  // Paired clients and pairing: confighttp_clients.cpp
  void getClients(resp_https_t response, req_https_t request);
  void updateClient(resp_https_t response, req_https_t request);
  void unpair(resp_https_t response, req_https_t request);
  void unpairAll(resp_https_t response, req_https_t request);
  void getOTP(resp_https_t response, req_https_t request);
  void savePin(resp_https_t response, req_https_t request);
  void disconnect(resp_https_t response, req_https_t request);

  // Configuration, logs and the host's lifetime: confighttp_config.cpp
  void getConfig(resp_https_t response, req_https_t request);
  void getLocale(resp_https_t response, req_https_t request);
  void getHost3dSubtitleApps(resp_https_t response, req_https_t request);
  void setHost3dSubtitleApp(resp_https_t response, req_https_t request);
  void saveConfig(resp_https_t response, req_https_t request);
  void getLogs(resp_https_t response, req_https_t request);
  void restart(resp_https_t response, req_https_t request);
  void quit(resp_https_t response, req_https_t request);

  // Convert Video: confighttp_offline_sbs.cpp
  void getOfflineSbsCapabilities(resp_https_t response, req_https_t request);
  void browseOfflineSbsFilesystem(resp_https_t response, req_https_t request);
  void listOfflineSbsJobs(resp_https_t response, req_https_t request);
  void createOfflineSbsJob(resp_https_t response, req_https_t request);
  void getOfflineSbsJob(resp_https_t response, req_https_t request);
  void cancelOfflineSbsJob(resp_https_t response, req_https_t request);
  void clearOfflineSbsJob(resp_https_t response, req_https_t request);

  // Game 3D setup: confighttp_game3d.cpp
  void getGame3dSetup(resp_https_t response, req_https_t request);
  void decideGame3dSetup(resp_https_t response, req_https_t request);
  /** The setup card as the Web UI reads it. */
  nlohmann::json game3d_setup_json(const platf::game3d::setup_status_t &status);

  // Local 3D displays: confighttp_local_display.cpp
  void getArGlassDevices(resp_https_t response, req_https_t request);
  void setArGlassDevice(resp_https_t response, req_https_t request);
  void getLocalMonitor(resp_https_t response, req_https_t request);
  void setLocalMonitor(resp_https_t response, req_https_t request);
}  // namespace confighttp

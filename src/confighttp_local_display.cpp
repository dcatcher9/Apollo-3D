/**
 * @file src/confighttp_local_display.cpp
 * @brief Web UI routes for local 3D displays: AR glasses and the shutter monitor.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

// standard includes
#include <sstream>
#include <string>

// lib includes
#include <nlohmann/json.hpp>

// local includes
#include "config.h"
#include "confighttp_common.h"
#include "confighttp_routes.h"

#include "platform/windows/ar_glasses.h"
#include "platform/windows/open3d_emitter.h"

namespace confighttp {
  void getArGlassDevices(resp_https_t response, req_https_t request) {
    nlohmann::json output_tree;
    output_tree["status"] = true;
    output_tree["devices"] = nlohmann::json::array();
    for (const auto &device : ar_glasses::devices()) {
      const char *decision = "pending";
      if (device.decision == ar_glasses::device_decision_e::approved) {
        decision = "approved";
      } else if (device.decision == ar_glasses::device_decision_e::rejected) {
        decision = "rejected";
      }
      output_tree["devices"].push_back({
        {"id", device.id},
        {"name", device.name},
        {"decision", decision},
        {"connected", device.connected},
        {"autoDetected", device.auto_detected},
      });
    }
    send_response(response, output_tree);
  }

  void getLocalMonitor(resp_https_t response, req_https_t request) {
    nlohmann::json output {
      {"status", true}, {"surface", config::local_surface_name(config::video.local_surface)},
      {"enabled", config::video.local_surface != config::video_t::local_surface_e::off},
      {"available", ar_glasses::local_surface_available()},
      {"source", ar_glasses::local_source_is_game() ? "game" : "host"},
      {"requested", ar_glasses::local_session_requested()}, {"active", ar_glasses::local_session_active()},
      {"error", ar_glasses::monitor_error()},
      {"timing_limits", nlohmann::json::array()},
    };
    // Session state above comes from memory and is safe to read often. Asking Windows for its
    // displays and serial ports is not, so that inventory is returned only when it is asked for:
    // when the settings page opens, on its Refresh button, and before saving.
    bool inventory = false;
    for (const auto &[name, value] : request->parse_query_string()) {
      inventory = inventory || (name == "inventory" && value == "1");
    }
    {
      // The settings this process is actually running with, in configuration-file form. Saving
      // without restarting changes the file but not these, and Start uses these.
      const auto &running = config::video.local_monitor;
      const auto flag = [](bool value) {
        return value ? "enabled" : "disabled";
      };
      output["running"] = {
        {"local_display_surface", config::local_surface_name(config::video.local_surface)},
        {"local_monitor_display_id", running.display_id},
        {"local_monitor_mode", ar_glasses::monitor_source_is_game() ? "game" : "host"},
        {"local_monitor_emitter_port", running.emitter_port},
        {"local_monitor_ir_protocol", running.ir_protocol},
        {"local_monitor_frame_delay_us", running.frame_delay_us},
        {"local_monitor_frame_duration_us", running.frame_duration_us},
        {"local_monitor_signal_spacing_us", running.signal_spacing_us},
        {"local_monitor_flip_eyes", flag(running.flip_eyes)},
        {"local_monitor_eye_repeats", running.eye_repeats},
        {"local_monitor_emitter_averaging", flag(running.emitter_averaging)},
      };
    }
    for (const int protocol : platf::open3d::limits::supported_protocols) {
      const auto limits = platf::open3d::timing_limits(protocol, 30);
      if (!limits) {
        continue;
      }
      output["timing_limits"].push_back({
        {"protocol", protocol},
        {"min_delay_us", limits->min_delay_us}, {"max_delay_us", limits->max_delay_us},
        {"min_duration_base_us", limits->min_duration_base_us}, {"min_duration_us", limits->min_duration_us},
        {"max_duration_us", limits->max_duration_us},
        {"min_signal_spacing_us", limits->min_signal_spacing_us}, {"max_signal_spacing_us", limits->max_signal_spacing_us},
        {"max_window_us", limits->max_window_us}, {"min_period_tail_base_us", limits->min_period_tail_base_us},
      });
    }
    if (inventory) {
      output["ports"] = platf::open3d::serial_ports();
      output["devices"] = nlohmann::json::array();
      for (const auto &device : ar_glasses::monitor_devices()) {
        output["devices"].push_back({{"id", device.id}, {"name", device.name},
          {"width", device.width}, {"height", device.height},
          {"refresh_millihz", device.refresh_millihz}, {"is_primary", device.is_primary}});
      }
    }
    send_response(response, output);
  }

  void setLocalMonitor(resp_https_t response, req_https_t request) {
    try {
      std::stringstream body;
      body << request->content.rdbuf();
      const auto input = nlohmann::json::parse(body);
      const auto action = input.at("action").get<std::string>();
      if (action == "source") {
        const auto source = input.at("source").get<std::string>();
        if ((source != "host" && source != "game") || !ar_glasses::set_local_source(source == "game")) {
          bad_request(response, request, "Choose Host 3D or Game 3D for a local display that is turned on.");
          return;
        }
        send_response(response, nlohmann::json {{"status", true}, {"source", source}});
        return;
      }
      if (action != "start" && action != "stop") {
        bad_request(response, request, "Choose start or stop for local 3D.");
        return;
      }
      if (!ar_glasses::request_local_session(action == "start")) {
        const auto error = ar_glasses::monitor_error();
        bad_request(response, request, error.empty() ? "Choose and configure a local display, save, and restart before starting 3D." : error);
        return;
      }
      send_response(response, nlohmann::json {{"status", true}, {"requested", ar_glasses::local_session_requested()}});
    } catch (const std::exception &error) {
      bad_request(response, request, error.what());
    }
  }

  void setArGlassDevice(resp_https_t response, req_https_t request) {
    try {
      const auto input = nlohmann::json::parse(request->content);
      const auto id = input.value("id", "");
      const auto decision_name = input.value("decision", "");
      const auto decision = decision_name == "approved" ?
                              ar_glasses::device_decision_e::approved :
                            decision_name == "rejected" ?
                              ar_glasses::device_decision_e::rejected :
                              ar_glasses::device_decision_e::pending;
      if (id.empty() || decision == ar_glasses::device_decision_e::pending || !ar_glasses::set_device_decision(id, decision)) {
        bad_request(response, request, "Unknown monitor or invalid AR display decision");
        return;
      }
      send_response(response, nlohmann::json {{"status", true}});
    } catch (const std::exception &error) {
      bad_request(response, request, error.what());
    }
  }
}  // namespace confighttp

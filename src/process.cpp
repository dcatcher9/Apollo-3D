/**
 * @file src/process.cpp
 * @brief Definitions for the startup and shutdown of the apps started by a streaming Session.
 */
#define BOOST_BIND_GLOBAL_PLACEHOLDERS

#ifndef BOOST_PROCESS_VERSION
  #define BOOST_PROCESS_VERSION 1
#endif
// standard includes
#include <condition_variable>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// lib includes
#include <boost/algorithm/string.hpp>
#include <boost/crc.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/token_functions.hpp>
#include <openssl/evp.h>
#include <openssl/sha.h>

// local includes
#include "config.h"
#include "crypto.h"
#include "display_device.h"
#include "file_handler.h"
#include "httpcommon.h"
#include "logging.h"
#include "platform/common.h"
#include "process.h"
#include "system_tray.h"
#include "utility.h"
#include "uuid.h"
#include "video.h"

#ifdef _WIN32
  #include "platform/windows/ar_glasses.h"
  // from_utf8() string conversion function
  #include "platform/windows/misc.h"
  #include "platform/windows/utils.h"

  // _SH constants for _wfsopen()
  #include <share.h>
#endif

#define DEFAULT_APP_IMAGE_PATH SUNSHINE_ASSETS_DIR "/box.png"

namespace proc {
  using namespace std::literals;
  namespace pt = boost::property_tree;

  proc_t proc;

  namespace {
    // Process/app state is shared by NVHTTP launch/resume threads, stream teardown, and the
    // local-AR topology controller. A recursive mutex is required because running() can perform
    // the same authoritative cleanup as terminate().
    std::recursive_mutex process_state_mutex;
#ifdef _WIN32
    // Topology removal outlives the proc_t configuration object that initiated it. Keep the
    // stable driver identity here so refresh() cannot discard it while replacing proc.
    std::optional<SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT> retired_virtual_display_identity;
    std::optional<GUID> retired_virtual_display_guid;
    std::wstring retired_virtual_display_device_path;
    std::wstring retired_virtual_display_gdi_name;
    bool retired_virtual_display_was_published = false;
    std::chrono::steady_clock::time_point retired_virtual_display_started {};
#endif
  }  // namespace

#ifdef _WIN32
  struct hdr_worker_state_t {
    std::mutex mutex;
    std::condition_variable_any display_changed;
    std::string display_name;
    bool desired_hdr {};
    std::uint64_t requested_revision {};
    std::uint64_t completed_revision {};
    bool completion_succeeded {};
    std::unordered_map<std::string, bool> original_hdr_states;
  };
#endif

#ifdef _WIN32
  std::atomic<VDISPLAY::DRIVER_STATUS> vDisplayDriverStatus {VDISPLAY::DRIVER_STATUS::UNKNOWN};

  void onVDisplayWatchdogFailed() {
    vDisplayDriverStatus.store(VDISPLAY::DRIVER_STATUS::WATCHDOG_FAILED);
  }

  void initVDisplayDriver() {
    vDisplayDriverStatus.store(VDISPLAY::openVDisplayDevice());
    if (vDisplayDriverStatus.load() == VDISPLAY::DRIVER_STATUS::OK) {
      if (!VDISPLAY::startPingThread(onVDisplayWatchdogFailed)) {
        VDISPLAY::closeVDisplayDevice();
        onVDisplayWatchdogFailed();
        return;
      }
    }
  }
#endif

  class deinit_t: public platf::deinit_t {
  public:
    ~deinit_t() {
      proc.terminate(false, false);
    }
  };

  std::unique_ptr<platf::deinit_t> init() {
    return std::make_unique<deinit_t>();
  }

  void terminate_process_group(boost::process::v1::child &proc, boost::process::v1::group &group, std::chrono::seconds exit_timeout) {
    if (group.valid() && platf::process_group_running((std::uintptr_t) group.native_handle())) {
      if (exit_timeout.count() > 0) {
        // Request processes in the group to exit gracefully
        if (platf::request_process_group_exit((std::uintptr_t) group.native_handle())) {
          // If the request was successful, wait for a little while for them to exit.
          BOOST_LOG(info) << "Successfully requested the app to exit. Waiting up to "sv << exit_timeout.count() << " seconds for it to close."sv;

          // group::wait_for() and similar functions are broken and deprecated, so we use a simple polling loop
          while (platf::process_group_running((std::uintptr_t) group.native_handle()) && (--exit_timeout).count() >= 0) {
            std::this_thread::sleep_for(1s);
          }

          if (exit_timeout.count() < 0) {
            BOOST_LOG(warning) << "App did not fully exit within the timeout. Terminating the app's remaining processes."sv;
          } else {
            BOOST_LOG(info) << "All app processes have successfully exited."sv;
          }
        } else {
          BOOST_LOG(info) << "App did not respond to a graceful termination request. Forcefully terminating the app's processes."sv;
        }
      } else {
        BOOST_LOG(info) << "No graceful exit timeout was specified for this app. Forcefully terminating the app's processes."sv;
      }

      // We always call terminate() even if we waited successfully for all processes above.
      // This ensures the process group state is consistent with the OS in boost.
      std::error_code ec;
      group.terminate(ec);
      group.detach();
    }

    if (proc.valid()) {
      // avoid zombie process
      proc.detach();
    }
  }

  boost::filesystem::path find_working_directory(const std::string &cmd, const boost::process::v1::environment &env) {
    // Parse the raw command string into parts to get the actual command portion
    std::vector<std::string> parts;
    try {
#ifdef _WIN32
      parts = boost::program_options::split_winmain(cmd);
#else
      parts = boost::program_options::split_unix(cmd);
#endif
    } catch (const boost::escaped_list_error &err) {
      BOOST_LOG(error) << "Boost failed to parse command ["sv << cmd << "]: "sv << err.what();
      return {};
    }

    if (parts.empty()) {
      BOOST_LOG(error) << "Unable to parse command: "sv << cmd;
      return boost::filesystem::path();
    }

    BOOST_LOG(debug) << "Parsed target ["sv << parts.at(0) << "] from command ["sv << cmd << ']';

    // If the target is a URL, don't parse any further here
    if (parts.at(0).find("://") != std::string::npos) {
      return boost::filesystem::path();
    }

    // If the cmd path is not an absolute path, resolve it using our PATH variable
    boost::filesystem::path cmd_path(parts.at(0));
    if (!cmd_path.is_absolute()) {
      cmd_path = boost::process::v1::search_path(parts.at(0));
      if (cmd_path.empty()) {
        BOOST_LOG(error) << "Unable to find executable ["sv << parts.at(0) << "]. Is it in your PATH?"sv;
        return boost::filesystem::path();
      }
    }

    BOOST_LOG(debug) << "Resolved target ["sv << parts.at(0) << "] to path ["sv << cmd_path << ']';

    // Now that we have a complete path, we can just use parent_path()
    return cmd_path.parent_path();
  }

  std::optional<render_size_t> calculate_render_size(
    std::uint32_t client_width,
    std::uint32_t client_height,
    std::uint32_t scale_factor
  ) {
    if (client_width == 0 || client_height == 0 || scale_factor == 0) {
      return std::nullopt;
    }

    const auto scaled_width = static_cast<std::uint64_t>(client_width) * scale_factor / 100;
    const auto scaled_height = static_cast<std::uint64_t>(client_height) * scale_factor / 100;
    if (scaled_width > std::numeric_limits<std::uint32_t>::max() || scaled_height > std::numeric_limits<std::uint32_t>::max()) {
      return std::nullopt;
    }

    // D3D/NVENC surfaces require even dimensions. Use the same floor behavior for every scale,
    // including 100%, so a reconnect cannot silently change the render-size contract.
    const auto width = static_cast<std::uint32_t>(scaled_width) & ~1U;
    const auto height = static_cast<std::uint32_t>(scaled_height) & ~1U;
    if (width == 0 || height == 0) {
      return std::nullopt;
    }
    return render_size_t {width, height};
  }

  std::string proc_t::get_display_name() const {
    std::lock_guard lock(process_state_mutex);
    return display_name;
  }

  void proc_t::set_display_name(std::string name) {
    std::lock_guard lock(process_state_mutex);
    set_display_name_locked(std::move(name));
  }

  void proc_t::set_display_name_locked(std::string name) {
    display_name = std::move(name);
#ifdef _WIN32
    if (_hdr_worker_state) {
      bool changed = false;
      {
        std::lock_guard state_lock(_hdr_worker_state->mutex);
        if (_hdr_worker_state->display_name != display_name) {
          _hdr_worker_state->display_name = display_name;
          ++_hdr_worker_state->requested_revision;
          changed = true;
        }
      }
      if (changed) {
        _hdr_worker_state->display_changed.notify_all();
      }
    }
#endif
  }

  process_status_t proc_t::get_status() {
    std::lock_guard lock(process_state_mutex);
    // This method is used by /serverinfo and launch admission while capture may still be active.
    // Never perform teardown from a read-only status request: the control/RTSP lifecycle observes
    // command exit, stops and joins the media workers, and only then calls running()/terminate().
    return {
      _app_id,
      _app_name,
      _app.uuid,
      _virtual_display,
      _host_session_id,
    };
  }

#ifdef _WIN32
  void proc_t::stop_hdr_worker() {
    _hdr_worker.request_stop();
    if (_hdr_worker_state) {
      _hdr_worker_state->display_changed.notify_all();
    }
    if (_hdr_worker.joinable()) {
      _hdr_worker.join();
    }
  }

  void proc_t::start_hdr_worker(bool enable_hdr) {
    stop_hdr_worker();

    auto state = std::make_shared<hdr_worker_state_t>();
    state->display_name = display_name;
    state->desired_hdr = enable_hdr;
    state->requested_revision = 1;
    _hdr_worker_state = state;
    _hdr_worker = std::jthread([state = std::move(state)](std::stop_token stop_token) {
      while (!stop_token.stop_requested()) {
        std::string current_display;
        bool desired_hdr = false;
        std::uint64_t revision = 0;
        {
          std::unique_lock lock(state->mutex);
          state->display_changed.wait(lock, stop_token, [&]() {
            return state->requested_revision > state->completed_revision && !state->display_name.empty();
          });
          if (stop_token.stop_requested()) {
            return;
          }
          current_display = state->display_name;
          desired_hdr = state->desired_hdr;
          revision = state->requested_revision;
        }

        const auto deadline = std::chrono::steady_clock::now() + 4s;
        auto request_was_superseded = [&]() {
          std::lock_guard lock(state->mutex);
          return stop_token.stop_requested() || state->requested_revision != revision;
        };
        auto wait_before_retry = [&](std::chrono::milliseconds delay) {
          std::unique_lock lock(state->mutex);
          state->display_changed.wait_for(lock, stop_token, delay, [&]() {
            return state->requested_revision != revision;
          });
          return stop_token.stop_requested() || state->requested_revision != revision;
        };

        bool success = false;
        std::optional<bool> observed_hdr;
        while (!request_was_superseded() && std::chrono::steady_clock::now() < deadline) {
          if (is_changing_settings_going_to_fail()) {
            if (wait_before_retry(100ms)) {
              break;
            }
            continue;
          }

          const auto display_name_w = platf::from_utf8(current_display);
          observed_hdr = VDISPLAY::queryDisplayHDRByName(display_name_w.c_str());
          if (!observed_hdr) {
            if (wait_before_retry(100ms)) {
              break;
            }
            continue;
          }

          {
            std::lock_guard lock(state->mutex);
            if (state->requested_revision != revision) {
              break;
            }
            // Preserve the state from before the first launch-time mutation. Reconfigure requests
            // must never replace this baseline with the HDR state left by an earlier request.
            state->original_hdr_states.try_emplace(current_display, *observed_hdr);
          }

          if (*observed_hdr == desired_hdr) {
            success = true;
            break;
          }
          if (!VDISPLAY::setDisplayHDRByName(display_name_w.c_str(), desired_hdr)) {
            if (wait_before_retry(100ms)) {
              break;
            }
            continue;
          }

          // The API can accept the request before the topology reports the new state. Verify it
          // instead of acknowledging a warm resume with the wrong dynamic-range contract.
          if (wait_before_retry(100ms)) {
            break;
          }
          observed_hdr = VDISPLAY::queryDisplayHDRByName(display_name_w.c_str());
          if (observed_hdr && *observed_hdr == desired_hdr) {
            success = true;
            break;
          }
        }

        {
          std::lock_guard lock(state->mutex);
          if (state->requested_revision != revision) {
            continue;
          }
          state->completed_revision = revision;
          state->completion_succeeded = success;
        }
        state->display_changed.notify_all();
        if (!success) {
          BOOST_LOG(warning) << "Could not apply and verify HDR " << (desired_hdr ? "on" : "off")
                             << " for display " << current_display;
        }
      }
    });
  }

  bool proc_t::request_hdr_state(bool enable_hdr, std::chrono::milliseconds timeout) {
    if (!_hdr_worker_state || !_hdr_worker.joinable()) {
      return false;
    }

    std::uint64_t revision;
    {
      std::lock_guard lock(_hdr_worker_state->mutex);
      _hdr_worker_state->desired_hdr = enable_hdr;
      revision = ++_hdr_worker_state->requested_revision;
    }
    _hdr_worker_state->display_changed.notify_all();

    std::unique_lock lock(_hdr_worker_state->mutex);
    const bool completed = _hdr_worker_state->display_changed.wait_for(lock, timeout, [&]() {
      return _hdr_worker_state->completed_revision >= revision;
    });
    return completed && _hdr_worker_state->completed_revision == revision && _hdr_worker_state->completion_succeeded;
  }
#endif

  int proc_t::execute(
    const ctx_t &app,
    std::shared_ptr<rtsp_stream::launch_session_t> launch_session,
    bool probe_encoder
  ) {
    std::lock_guard lock(process_state_mutex);
    // Ensure starting from a clean slate.
    terminate(false, false);

    const uint32_t client_width = launch_session->width ? launch_session->width : 1920;
    const uint32_t client_height = launch_session->height ? launch_session->height : 1080;
    const auto render_size = calculate_render_size(client_width, client_height, launch_session->scale_factor);
    if (!render_size) {
      BOOST_LOG(error) << "Requested client mode and render scale produce an invalid render size."sv;
      return 400;
    }
    const uint32_t render_width = render_size->width;
    const uint32_t render_height = render_size->height;
    const int scale_factor = launch_session->scale_factor;

    // Do not publish process ownership until every side-effect-free request validation has passed.
    // In particular, a tiny scaled mode can floor to zero after enforcing even dimensions; that
    // must leave the host idle rather than advertising an app that was never launched.
    _app = app;
    _app_id = util::from_view(app.id);
    _app_name = app.name;
    _launch_session = launch_session;

    launch_session->width = render_width;
    launch_session->height = render_height;

    this->initial_display = config::video.output_name;
    // Executed when returning from function
    auto fg = util::fail_guard([&]() {
      // Restore to user defined output name
      config::video.output_name = this->initial_display;
      terminate();
    });

#ifdef _WIN32
    const bool needs_virtual_display =
      launch_session->virtual_display  // User requested virtual display
      || _app.synthetic_virtual_display  // Apollo's generated Virtual Display tile
      || !video::allow_encoder_probing();  // No active display presents
    if (needs_virtual_display) {
      if (!ar_glasses::remote_virtual_display_starting(
            launch_session->id,
            config::stream.ping_timeout,
            true
          )) {
        BOOST_LOG(error) << "Timed out waiting for local AR to release virtual-display ownership."sv;
        launch_session->virtual_display = false;
        return 503;
      }
      _remote_virtual_display_lease = launch_session->id;
      if (!wait_for_retired_virtual_display(3s)) {
        BOOST_LOG(error) << "Timed out waiting for the previous virtual display to leave the Windows topology."sv;
        launch_session->virtual_display = false;
        return 503;
      }
      if (vDisplayDriverStatus != VDISPLAY::DRIVER_STATUS::OK) {
        // Try init driver again
        initVDisplayDriver();
      }

      if (vDisplayDriverStatus == VDISPLAY::DRIVER_STATUS::OK) {
        std::string device_name;
        std::string device_uuid_str;
        uuid_util::uuid_t device_uuid;

        device_name = launch_session->device_name;
        device_uuid_str = launch_session->unique_id;
        device_uuid = uuid_util::uuid_t::parse(launch_session->unique_id);

        memcpy(&launch_session->display_guid, &device_uuid, sizeof(GUID));

        int target_fps = launch_session->fps ? launch_session->fps : 60000;

        if (target_fps < 1000) {
          target_fps *= 1000;
        }

        VDISPLAY::creation_result_t created_display;
        if (!config::video.adapter_name.empty()) {
          created_display = VDISPLAY::createVirtualDisplayOnAdapter(
            device_uuid_str.c_str(),
            device_name.c_str(),
            render_width,
            render_height,
            target_fps,
            launch_session->display_guid,
            platf::from_utf8(config::video.adapter_name)
          );
        } else {
          created_display = VDISPLAY::createVirtualDisplay(
            device_uuid_str.c_str(),
            device_name.c_str(),
            render_width,
            render_height,
            target_fps,
            launch_session->display_guid
          );
        }

        // Add success and GDI-name publication are separate. Track the exact driver identity only
        // when AddVirtualDisplay succeeded so a failed launch cannot remove a display it did not
        // create, while a name-timeout still receives deterministic teardown.
        launch_session->virtual_display = created_display.added();
        _virtual_display_identity = created_display.identity;
        _virtual_display_device_path = created_display.device_path;
        _virtual_display_gdi_name = created_display.display_name;
        _virtual_display_published = !created_display.display_name.empty();

        if (!created_display.display_name.empty()) {
          BOOST_LOG(info) << "Virtual Display created at " << created_display.display_name;

          // Don't change display settings when no params are given
          if (launch_session->width && launch_session->height && launch_session->fps) {
            // Apply display settings
            if (VDISPLAY::changeDisplaySettings(created_display.display_name.c_str(), render_width, render_height, target_fps) != DISP_CHANGE_SUCCESSFUL) {
              BOOST_LOG(error) << "Windows did not accept the requested virtual-display mode."sv;
              return 503;
            }
          }

          // Set virtual_display to true when everything went fine
          _virtual_display = true;
          set_display_name_locked(platf::to_utf8(created_display.display_name));

          // When using virtual display, we don't care which display user configured to use.
          // So we always set output_name to the newly created virtual display as a workaround for
          // empty name when probing graphics cards.

          config::video.output_name = display_device::map_display_name(this->display_name);
        } else {
          BOOST_LOG(error) << (created_display.added() ? "Virtual display was added, but Windows did not publish its display name in time." : "Virtual display creation failed.");
          ar_glasses::remote_virtual_display_ended(*_remote_virtual_display_lease);
          _remote_virtual_display_lease.reset();
          return 503;
        }
      } else {
        BOOST_LOG(error) << "A virtual display is required, but the SudoVDA driver is unavailable."sv;
        launch_session->virtual_display = false;
        ar_glasses::remote_virtual_display_ended(*_remote_virtual_display_lease);
        _remote_virtual_display_lease.reset();
        return 503;
      }
      if (!_virtual_display) {
        ar_glasses::remote_virtual_display_ended(*_remote_virtual_display_lease);
        _remote_virtual_display_lease.reset();
      }
    }

#endif

    // Probe encoders again before streaming to ensure our chosen
    // encoder matches the active GPU (which could have changed
    // due to hotplugging, driver crash, primary monitor change,
    // or any number of other factors).
    if (probe_encoder && video::probe_encoders()) {
      return 503;
    }

    std::string fps_str;
    char fps_buf[8];
    snprintf(fps_buf, sizeof(fps_buf), "%.3f", (float) launch_session->fps / 1000.0f);
    fps_str = fps_buf;

    // Add stream-specific environment variables for application and preparation commands.
    _env["APOLLO_APP_ID"] = _app.id;
    _env["APOLLO_APP_NAME"] = _app.name;
    _env["APOLLO_APP_UUID"] = _app.uuid;
    _env["APOLLO_APP_STATUS"] = "STARTING";
    _env["APOLLO_CLIENT_UUID"] = launch_session->unique_id;
    _env["APOLLO_CLIENT_NAME"] = launch_session->device_name;
    _env["APOLLO_CLIENT_WIDTH"] = std::to_string(render_width);
    _env["APOLLO_CLIENT_HEIGHT"] = std::to_string(render_height);
    _env["APOLLO_CLIENT_RENDER_WIDTH"] = std::to_string(launch_session->width);
    _env["APOLLO_CLIENT_RENDER_HEIGHT"] = std::to_string(launch_session->height);
    _env["APOLLO_CLIENT_SCALE_FACTOR"] = std::to_string(scale_factor);
    _env["APOLLO_CLIENT_FPS"] = fps_str;
    _env["APOLLO_CLIENT_HDR"] = launch_session->enable_hdr ? "true" : "false";
    _env["APOLLO_CLIENT_HOST_AUDIO"] = launch_session->host_audio ? "true" : "false";

    int channelCount = launch_session->surround_info & 65535;
    switch (channelCount) {
      case 2:
        _env["APOLLO_CLIENT_AUDIO_CONFIGURATION"] = "2.0";
        break;
      case 6:
        _env["APOLLO_CLIENT_AUDIO_CONFIGURATION"] = "5.1";
        break;
      case 8:
        _env["APOLLO_CLIENT_AUDIO_CONFIGURATION"] = "7.1";
        break;
    }

    if (!_app.output.empty() && _app.output != "null"sv) {
#ifdef _WIN32
      // fopen() interprets the filename as an ANSI string on Windows, so we must convert it
      // to UTF-16 and use the wchar_t variants for proper Unicode log file path support.
      auto woutput = platf::from_utf8(_app.output);

      // Use _SH_DENYNO to allow us to open this log file again for writing even if it is
      // still open from a previous execution. This is required to handle the case of a
      // detached process executing again while the previous process is still running.
      _pipe.reset(_wfsopen(woutput.c_str(), L"a", _SH_DENYNO));
#else
      _pipe.reset(fopen(_app.output.c_str(), "a"));
#endif
    }

    std::error_code ec;
    _completed_prep_commands = 0;

    for (std::size_t prep_index = 0; prep_index < _app.prep_cmds.size(); ++prep_index) {
      auto &cmd = _app.prep_cmds[prep_index];

      // Skip empty commands
      if (cmd.do_cmd.empty()) {
        _completed_prep_commands = prep_index + 1;
        continue;
      }

      boost::filesystem::path working_dir = _app.working_dir.empty() ?
                                              find_working_directory(cmd.do_cmd, _env) :
                                              boost::filesystem::path(_app.working_dir);
      BOOST_LOG(info) << "Executing Do Cmd: ["sv << cmd.do_cmd << "] elevated: " << cmd.elevated;
      auto child = platf::run_command(cmd.elevated, true, cmd.do_cmd, working_dir, _env, _pipe.get(), ec, nullptr);

      if (ec) {
        BOOST_LOG(error) << "Couldn't run ["sv << cmd.do_cmd << "]: System: "sv << ec.message();
        // We don't want any prep commands failing launch of the desktop.
        // This is to prevent the issue where users reboot their PC and need to log in with Sunshine.
        // permission_denied is typically returned when the user impersonation fails, which can happen when user is not signed in yet.
        if (!(_app.cmd.empty() && ec == std::errc::permission_denied)) {
          return -1;
        }

        // The desktop session is intentionally allowed to continue without running any later
        // prep commands if user impersonation is unavailable. Preserve the successful prefix so
        // teardown only runs undo commands corresponding to prep commands that actually ran.
        break;
      }

      child.wait(ec);
      if (ec) {
        BOOST_LOG(error) << '[' << cmd.do_cmd << "] wait failed: "sv << ec.message();
        return -1;
      }

      auto ret = child.exit_code();
      if (ret != 0) {
        BOOST_LOG(error) << '[' << cmd.do_cmd << "] exited with code ["sv << ret << ']';
        return -1;
      }
      _completed_prep_commands = prep_index + 1;
    }

    _env["APOLLO_APP_STATUS"] = "RUNNING";

    for (auto &cmd : _app.detached) {
      boost::filesystem::path working_dir = _app.working_dir.empty() ?
                                              find_working_directory(cmd, _env) :
                                              boost::filesystem::path(_app.working_dir);
      BOOST_LOG(info) << "Spawning ["sv << cmd << "] in ["sv << working_dir << ']';
      auto child = platf::run_command(_app.elevated, true, cmd, working_dir, _env, _pipe.get(), ec, nullptr);
      if (ec) {
        BOOST_LOG(warning) << "Couldn't spawn ["sv << cmd << "]: System: "sv << ec.message();
      } else {
        child.detach();
      }
    }

    if (_app.cmd.empty()) {
      BOOST_LOG(info) << "No commands configured, showing desktop..."sv;
      placebo = true;
    } else {
      boost::filesystem::path working_dir = _app.working_dir.empty() ?
                                              find_working_directory(_app.cmd, _env) :
                                              boost::filesystem::path(_app.working_dir);
      BOOST_LOG(info) << "Executing: ["sv << _app.cmd << "] in ["sv << working_dir << ']';
      _process = platf::run_command(_app.elevated, true, _app.cmd, working_dir, _env, _pipe.get(), ec, &_process_group);
      if (ec) {
        BOOST_LOG(warning) << "Couldn't run ["sv << _app.cmd << "]: System: "sv << ec.message();
        return -1;
      }
    }

    _app_launch_time = std::chrono::steady_clock::now();

    do {
      if (RAND_bytes(reinterpret_cast<unsigned char *>(&_host_session_id), sizeof(_host_session_id)) != 1) {
        BOOST_LOG(error) << "Failed to generate a retained host-session token."sv;
        return 500;
      }
    } while (_host_session_id == 0);

#ifdef _WIN32
    start_hdr_worker(launch_session->enable_hdr);
    if (_virtual_display && !request_hdr_state(launch_session->enable_hdr, 6s)) {
      BOOST_LOG(error) << "The virtual display did not reach the requested HDR state."sv;
      return 503;
    }
    if (_virtual_display) {
      // Virtual-display creation, encoder probing, and application prep can take longer than the
      // original client-connect reservation. Renew it while process_state_mutex is still held so
      // local AR cannot claim and destroy this display in the gap before RTSP calls resume().
      ar_glasses::remote_virtual_display_awaiting_client(
        *_remote_virtual_display_lease,
        config::stream.ping_timeout
      );
    }
#endif

    fg.disable();

#if defined SUNSHINE_TRAY && SUNSHINE_TRAY >= 1
    system_tray::update_tray_playing(_app.name);
#endif

    return 0;
  }

  int proc_t::running() {
    std::lock_guard lock(process_state_mutex);
    return running_locked();
  }

  bool proc_t::stream_process_exited() {
    std::lock_guard lock(process_state_mutex);
    return stream_process_exited_locked();
  }

  bool proc_t::stream_process_exited_locked() {
#ifndef _WIN32
    // On POSIX OSes, we must periodically wait for our children to avoid
    // them becoming zombies. This must be synchronized carefully with
    // calls to bp::wait() and platf::process_group_running() which both
    // invoke waitpid() under the hood.
    auto reaper = util::fail_guard([]() {
      while (waitpid(-1, nullptr, WNOHANG) > 0);
    });
#endif

    if (_app_id <= 0 || placebo) {
      return false;
    } else if (_app.wait_all && _process_group && platf::process_group_running((std::uintptr_t) _process_group.native_handle())) {
      return false;
    } else if (_process.running()) {
      return false;
    } else if (_app.auto_detach && std::chrono::steady_clock::now() - _app_launch_time < 5s) {
      BOOST_LOG(info) << "App exited with code ["sv << _process.native_exit_code() << "] within 5 seconds of launch. Treating the app as a detached command."sv;
      BOOST_LOG(info) << "Adjust this behavior in the Applications tab or apps.json if this is not what you want."sv;
      placebo = true;

#if defined SUNSHINE_TRAY && SUNSHINE_TRAY >= 1
      if (_process.native_exit_code() != 0) {
        system_tray::update_tray_launch_error(proc::proc.get_last_run_app_name(), _process.native_exit_code());
      }
#endif

      return false;
    }

    return static_cast<bool>(_process);
  }

  int proc_t::running_locked() {
    if (!stream_process_exited_locked()) {
      return _app_id;
    }

    // Perform cleanup actions now if needed
    terminate();

    return 0;
  }

  std::uint64_t proc_t::get_host_session_id() const {
    std::lock_guard lock(process_state_mutex);
    return _host_session_id;
  }

  int proc_t::reconfigure_retained_session(std::shared_ptr<rtsp_stream::launch_session_t> launch_session) {
    std::lock_guard lock(process_state_mutex);
    if (!launch_session || _app_id <= 0 || _host_session_id == 0 || !_launch_session) {
      return 409;
    }

    const auto render_size = calculate_render_size(
      launch_session->width,
      launch_session->height,
      launch_session->scale_factor
    );
    if (!render_size) {
      return 400;
    }

    // A retained physical-desktop process cannot be converted into a virtual-display process
    // without changing desktop ownership. A process whose virtual display was forced because no
    // physical output existed remains virtual even if the client again sends virtualDisplay=0.
    if (launch_session->virtual_display && !_virtual_display) {
      BOOST_LOG(warning) << "A warm resume cannot change a physical-desktop process into a virtual-display process."sv;
      return 409;
    }

    const auto old_width = static_cast<std::uint32_t>(_launch_session->width);
    const auto old_height = static_cast<std::uint32_t>(_launch_session->height);
    const auto old_fps = _launch_session->fps;
    const bool old_hdr = _launch_session->enable_hdr;
    const bool display_mode_changed = _virtual_display &&
                                      (old_width != render_size->width || old_height != render_size->height || old_fps != launch_session->fps);

#ifdef _WIN32
    if (_virtual_display) {
      if (_virtual_display_gdi_name.empty()) {
        BOOST_LOG(error) << "The retained virtual display no longer has a published Windows display name."sv;
        return 503;
      }
      if (display_mode_changed &&
          VDISPLAY::changeDisplaySettings(
            _virtual_display_gdi_name.c_str(),
            render_size->width,
            render_size->height,
            launch_session->fps
          ) != DISP_CHANGE_SUCCESSFUL) {
        // changeDisplaySettings() first applies a baseline mode before its exact fractional mode.
        // Roll back even on failure because that baseline call may already have changed topology.
        VDISPLAY::changeDisplaySettings(_virtual_display_gdi_name.c_str(), old_width, old_height, old_fps);
        BOOST_LOG(error) << "Failed to reconfigure the retained virtual-display mode."sv;
        return 503;
      }
    }

    if (!request_hdr_state(launch_session->enable_hdr, 6s)) {
      if (display_mode_changed &&
          VDISPLAY::changeDisplaySettings(_virtual_display_gdi_name.c_str(), old_width, old_height, old_fps) != DISP_CHANGE_SUCCESSFUL) {
        BOOST_LOG(error) << "Failed to roll back the retained virtual-display mode after HDR reconfiguration failed."sv;
      }
      if (!request_hdr_state(old_hdr, 6s)) {
        BOOST_LOG(error) << "Failed to restore the retained session's prior HDR state after reconfiguration failed."sv;
      }
      return 503;
    }
#endif

    // Commit only after every fallible display operation succeeds. The new object keeps its own
    // RTSP encryption and transport lease, while the process-owned launch record retains its
    // original identity/GUID and stable host-session token.
    launch_session->width = render_size->width;
    launch_session->height = render_size->height;
    launch_session->virtual_display = _virtual_display;
#ifdef _WIN32
    launch_session->display_guid = _launch_session->display_guid;
#endif
    _launch_session->width = launch_session->width;
    _launch_session->height = launch_session->height;
    _launch_session->fps = launch_session->fps;
    _launch_session->enable_hdr = launch_session->enable_hdr;
    _launch_session->scale_factor = launch_session->scale_factor;
    _launch_session->sbs_mode = launch_session->sbs_mode;
    return 0;
  }

  bool proc_t::activate_remote_virtual_display_lease(std::uint64_t lease) {
    std::lock_guard lock(process_state_mutex);
#ifdef _WIN32
    if (_launch_session && _launch_session->virtual_display) {
      if (lease == 0 || !ar_glasses::remote_virtual_display_active(lease)) {
        BOOST_LOG(error) << "Remote virtual-display activation did not match its ownership lease."sv;
        return false;
      }
      _remote_virtual_display_lease = lease;
    }
#endif
    return true;
  }

  void proc_t::terminate(bool immediate, bool needs_refresh) {
    std::lock_guard lock(process_state_mutex);
#ifdef _WIN32
    // The worker never takes process_state_mutex, so it is safe to join while holding the process
    // state lock. This prevents an old launch from touching a display after teardown or refresh.
    stop_hdr_worker();
    std::unordered_map<std::string, bool> original_hdr_states;
    if (_hdr_worker_state) {
      std::lock_guard state_lock(_hdr_worker_state->mutex);
      original_hdr_states = _hdr_worker_state->original_hdr_states;
    }
#endif
    std::error_code ec;
    placebo = false;

    if (!immediate) {
      terminate_process_group(_process, _process_group, _app.exit_timeout);
    }

    _process = boost::process::v1::child();
    _process_group = boost::process::v1::group();

    _env["APOLLO_APP_STATUS"] = "TERMINATING";

    while (_completed_prep_commands > 0) {
      auto &cmd = _app.prep_cmds[--_completed_prep_commands];

      if (cmd.undo_cmd.empty()) {
        continue;
      }

      boost::filesystem::path working_dir = _app.working_dir.empty() ?
                                              find_working_directory(cmd.undo_cmd, _env) :
                                              boost::filesystem::path(_app.working_dir);
      BOOST_LOG(info) << "Executing Undo Cmd: ["sv << cmd.undo_cmd << ']';
      auto child = platf::run_command(cmd.elevated, true, cmd.undo_cmd, working_dir, _env, _pipe.get(), ec, nullptr);

      if (ec) {
        BOOST_LOG(warning) << "System: "sv << ec.message();
        continue;
      }

      child.wait(ec);
      if (ec) {
        BOOST_LOG(warning) << '[' << cmd.undo_cmd << "] wait failed: "sv << ec.message();
        continue;
      }

      auto ret = child.exit_code();

      if (ret != 0) {
        BOOST_LOG(warning) << "Return code ["sv << ret << ']';
      }
    }

    _pipe.reset();

    bool has_run = _app_id > 0;

#ifdef _WIN32
    if (_remote_virtual_display_lease) {
      ar_glasses::remote_virtual_display_ended(*_remote_virtual_display_lease);
      _remote_virtual_display_lease.reset();
    }

    // Revert HDR state
    if (has_run) {
      for (const auto &[changed_display, initial_hdr] : original_hdr_states) {
        const auto display_name_w = platf::from_utf8(changed_display);
        const auto restore_deadline = std::chrono::steady_clock::now() + 2s;
        bool restored = false;
        while (std::chrono::steady_clock::now() < restore_deadline) {
          if (is_changing_settings_going_to_fail()) {
            std::this_thread::sleep_for(100ms);
            continue;
          }

          const auto current_hdr = VDISPLAY::queryDisplayHDRByName(display_name_w.c_str());
          if (current_hdr && *current_hdr == initial_hdr) {
            restored = true;
            break;
          }

          VDISPLAY::setDisplayHDRByName(display_name_w.c_str(), initial_hdr);
          std::this_thread::sleep_for(100ms);
        }
        if (!restored) {
          const auto current_hdr = VDISPLAY::queryDisplayHDRByName(display_name_w.c_str());
          restored = current_hdr && *current_hdr == initial_hdr;
        }

        if (restored) {
          BOOST_LOG(info) << "HDR reverted for display " << changed_display;
        } else {
          BOOST_LOG(error) << "HDR revert could not be verified for display " << changed_display;
        }
      }
    }

    bool used_virtual_display = vDisplayDriverStatus == VDISPLAY::DRIVER_STATUS::OK && _launch_session && _launch_session->virtual_display;
    if (used_virtual_display) {
      if (_virtual_display_identity) {
        retired_virtual_display_identity = _virtual_display_identity;
        retired_virtual_display_guid = _launch_session->display_guid;
        retired_virtual_display_device_path = _virtual_display_device_path;
        retired_virtual_display_gdi_name = _virtual_display_gdi_name;
        retired_virtual_display_was_published = _virtual_display_published;
        retired_virtual_display_started = std::chrono::steady_clock::now();
      }
      if (VDISPLAY::removeVirtualDisplay(_launch_session->display_guid)) {
        BOOST_LOG(info) << "Virtual Display removed successfully";
      } else if (_virtual_display) {
        BOOST_LOG(warning) << "Virtual Display remove failed";
      } else {
        BOOST_LOG(warning) << "Virtual Display remove failed, but it seems it was not created correctly either.";
      }
    }
#endif

    // Only show the Stopped notification if we actually have an app to stop
    // Since terminate() is always run when a new app has started
    if (proc::proc.get_last_run_app_name().length() > 0 && has_run) {
#if defined SUNSHINE_TRAY && SUNSHINE_TRAY >= 1
      system_tray::update_tray_stopped(proc::proc.get_last_run_app_name());
#endif
    }

    // Load the configured output_name first
    // to prevent the value being write to empty when the initial terminate happens
    if (!has_run && initial_display.empty()) {
      initial_display = config::video.output_name;
    } else {
      // Restore output name to its original value
      config::video.output_name = initial_display;
    }

    _app_id = 0;
    _app_name.clear();
    _app = {};
    _completed_prep_commands = 0;
    _host_session_id = 0;
    display_name.clear();
    initial_display.clear();
    _launch_session.reset();
#ifdef _WIN32
    _virtual_display_identity.reset();
    _virtual_display_device_path.clear();
    _virtual_display_gdi_name.clear();
    _virtual_display_published = false;
    _remote_virtual_display_lease.reset();
    _hdr_worker_state.reset();
#endif
    _virtual_display = false;

    if (needs_refresh) {
      refresh(config::stream.file_apps, false);
    }
  }

#ifdef _WIN32
  bool proc_t::wait_for_retired_virtual_display(std::chrono::milliseconds timeout) {
    if (!retired_virtual_display_identity) {
      return true;
    }

    const auto identity = *retired_virtual_display_identity;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    constexpr unsigned int required_absent_observations = 3;
    unsigned int consecutive_absent_observations = 0;
    auto next_remove_retry = std::chrono::steady_clock::now();

    auto is_same_identity = [&](const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &candidate) {
      return candidate.AdapterLuid.HighPart == identity.AdapterLuid.HighPart &&
             candidate.AdapterLuid.LowPart == identity.AdapterLuid.LowPart &&
             candidate.TargetId == identity.TargetId;
    };

    // Always make one immediate observation, even for a zero timeout. A failed topology query is
    // indeterminate, never evidence that the retired output disappeared.
    while (true) {
      if (!retired_virtual_display_identity) {
        return true;
      }
      if (!is_same_identity(*retired_virtual_display_identity)) {
        return false;
      }

      const auto observation_time = std::chrono::steady_clock::now();
      if (retired_virtual_display_guid && observation_time >= next_remove_retry) {
        // Driver removal can fail transiently during an IddCx/topology transition. Retain the GUID
        // with the stable identity and retry; otherwise one failed IOCTL blocks all later virtual
        // displays until Apollo restarts.
        VDISPLAY::removeVirtualDisplay(*retired_virtual_display_guid);
        next_remove_retry = observation_time + 250ms;
      }

      const auto query = VDISPLAY::queryVirtualDisplayIdentity(
        identity,
        retired_virtual_display_device_path,
        retired_virtual_display_gdi_name
      );
      if (query.state == VDISPLAY::display_identity_state_e::absent) {
        ++consecutive_absent_observations;
        if (consecutive_absent_observations >= required_absent_observations) {
          // An AddVirtualDisplay result can precede Windows publishing the path. For such a display,
          // quarantine the stable identity for one topology debounce even when every snapshot says
          // absent, so a delayed arrival/removal notification cannot race the replacement display.
          if (!retired_virtual_display_was_published && std::chrono::steady_clock::now() - retired_virtual_display_started < 750ms) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
              return false;
            }
            std::this_thread::sleep_for(std::min(50ms, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)));
            continue;
          }
          if (std::chrono::steady_clock::now() + 250ms > deadline) {
            return false;
          }
          std::this_thread::sleep_for(250ms);
          const auto settled_query = VDISPLAY::queryVirtualDisplayIdentity(
            identity,
            retired_virtual_display_device_path,
            retired_virtual_display_gdi_name
          );
          if (settled_query.state != VDISPLAY::display_identity_state_e::absent) {
            consecutive_absent_observations = 0;
            continue;
          }
          // Never clear a newer retirement record that replaced the one this wait observed.
          if (!retired_virtual_display_identity || !is_same_identity(*retired_virtual_display_identity)) {
            return false;
          }
          retired_virtual_display_identity.reset();
          retired_virtual_display_guid.reset();
          retired_virtual_display_device_path.clear();
          retired_virtual_display_gdi_name.clear();
          retired_virtual_display_was_published = false;
          retired_virtual_display_started = {};
          return !retired_virtual_display_identity.has_value();
        }
      } else {
        consecutive_absent_observations = 0;
      }

      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(std::min(50ms, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)));
    }
  }

  local_ar_handoff_e proc_t::prepare_local_ar_handoff(const std::stop_source &construction_stop) {
    std::lock_guard lock(process_state_mutex);

    // Recheck after acquiring the process lock. A remote launch may have reserved ownership
    // after the topology controller's first observation but before it reached this method.
    if (ar_glasses::remote_virtual_display_blocks_local()) {
      return local_ar_handoff_e::remote_busy;
    }
    if (!ar_glasses::wait_for_local_virtual_display_retirement(3s)) {
      BOOST_LOG(warning) << "Timed out waiting for the retired local AR virtual display to leave the Windows topology."sv;
      return local_ar_handoff_e::cleanup_timeout;
    }
    if (!wait_for_retired_virtual_display(3s)) {
      BOOST_LOG(warning) << "Timed out waiting for the retired remote virtual display to leave the Windows topology."sv;
      return local_ar_handoff_e::cleanup_timeout;
    }
    // Claim before terminating an inactive retained process. remote_virtual_display_starting()
    // does not take process_state_mutex, so a check followed by termination is not atomic. This
    // ownership claim is the serialization point: a reconnect that reserved first wins, while a
    // later reconnect waits for this bounded local construction to release or cancel.
    if (!ar_glasses::try_claim_local_virtual_display(construction_stop)) {
      return local_ar_handoff_e::remote_busy;
    }
    auto release_claim = util::fail_guard([]() {
      ar_glasses::release_local_virtual_display_claim();
    });
    if (construction_stop.stop_requested()) {
      return local_ar_handoff_e::remote_busy;
    }
    if (!_launch_session || !_launch_session->virtual_display) {
      release_claim.disable();
      return local_ar_handoff_e::ready;
    }

    const auto app_name = _app_name;
    BOOST_LOG(info) << "Local AR is taking ownership from inactive remote virtual-display session ["sv
                    << app_name << "]."sv;
    terminate(false, false);

    // SudoVDA acknowledges removal before Windows necessarily publishes the new topology. Wait
    // on the stable adapter/target identity, not a transient DISPLAY number, before adding the
    // local source. This prevents the old removal notification from deleting/renumbering the new
    // display underneath its presenter. Retain timed-out identities so a retry cannot forget them.
    if (retired_virtual_display_identity) {
      if (wait_for_retired_virtual_display(3s)) {
        release_claim.disable();
        return local_ar_handoff_e::ready;
      }
      BOOST_LOG(warning) << "Timed out waiting for the inactive remote virtual display to leave the Windows topology."sv;
      return local_ar_handoff_e::cleanup_timeout;
    }

    // Older or failed creations may not have a driver identity. Termination is still synchronous
    // at the driver boundary, so allow one topology-debounce interval before local creation.
    std::this_thread::sleep_for(750ms);
    if (construction_stop.stop_requested()) {
      return local_ar_handoff_e::remote_busy;
    }
    release_claim.disable();
    return local_ar_handoff_e::ready;
  }
#endif

  std::vector<ctx_t> proc_t::get_apps() const {
    std::lock_guard lock(process_state_mutex);
    return _apps;
  }

  // Gets application image from application list.
  // Returns image from assets directory if found there.
  // Returns default image if image configuration is not set.
  // Returns http content-type header compatible image type.
  std::string proc_t::get_app_image(int app_id) {
    std::lock_guard lock(process_state_mutex);
    auto iter = std::find_if(_apps.begin(), _apps.end(), [&app_id](const auto app) {
      return app.id == std::to_string(app_id);
    });
    auto app_image_path = iter == _apps.end() ? std::string() : iter->image_path;

    return validate_app_image_path(app_image_path);
  }

  std::string proc_t::get_last_run_app_name() {
    std::lock_guard lock(process_state_mutex);
    return _app_name;
  }

  std::string proc_t::get_running_app_uuid() {
    std::lock_guard lock(process_state_mutex);
    return _app.uuid;
  }

  proc_t::~proc_t() {
    // It's not safe to call terminate() here because our proc_t is a static variable
    // that may be destroyed after the Boost loggers have been destroyed. Instead,
    // we return a deinit_t to main() to handle termination when we're exiting.
    // Once we reach this point here, termination must have already happened.
    assert(!placebo);
    assert(!_process.running());
#ifdef _WIN32
    assert(!_hdr_worker.joinable());
#endif
  }

  std::string_view::iterator find_match(std::string_view::iterator begin, std::string_view::iterator end) {
    int stack = 0;

    --begin;
    do {
      ++begin;
      switch (*begin) {
        case '(':
          ++stack;
          break;
        case ')':
          --stack;
      }
    } while (begin != end && stack != 0);

    if (begin == end) {
      throw std::out_of_range("Missing closing bracket \')\'");
    }
    return begin;
  }

  std::string parse_env_val(boost::process::v1::native_environment &env, const std::string_view &val_raw) {
    auto pos = std::begin(val_raw);
    auto dollar = std::find(pos, std::end(val_raw), '$');

    std::stringstream ss;

    while (dollar != std::end(val_raw)) {
      auto next = dollar + 1;
      if (next != std::end(val_raw)) {
        switch (*next) {
          case '(':
            {
              ss.write(pos, (dollar - pos));
              auto var_begin = next + 1;
              auto var_end = find_match(next, std::end(val_raw));
              auto var_name = std::string {var_begin, var_end};

#ifdef _WIN32
              // Windows treats environment variable names in a case-insensitive manner,
              // so we look for a case-insensitive match here. This is critical for
              // correctly appending to PATH on Windows.
              auto itr = std::find_if(env.cbegin(), env.cend(), [&](const auto &e) {
                return boost::iequals(e.get_name(), var_name);
              });
              if (itr != env.cend()) {
                // Use an existing case-insensitive match
                var_name = itr->get_name();
              }
#endif

              ss << env[var_name].to_string();

              pos = var_end + 1;
              next = var_end;

              break;
            }
          case '$':
            ss.write(pos, (next - pos));
            pos = next + 1;
            ++next;
            break;
        }

        dollar = std::find(next, std::end(val_raw), '$');
      } else {
        dollar = next;
      }
    }

    ss.write(pos, (dollar - pos));

    return ss.str();
  }

  std::string validate_app_image_path(std::string app_image_path) {
    if (app_image_path.empty()) {
      return DEFAULT_APP_IMAGE_PATH;
    }

    // get the image extension and convert it to lowercase
    auto image_extension = std::filesystem::path(app_image_path).extension().string();
    boost::to_lower(image_extension);

    // return the default box image if extension is not "png"
    if (image_extension != ".png") {
      return DEFAULT_APP_IMAGE_PATH;
    }

    // check if image is in assets directory
    auto full_image_path = std::filesystem::path(SUNSHINE_ASSETS_DIR) / app_image_path;
    if (std::filesystem::exists(full_image_path)) {
      return full_image_path.string();
    } else if (app_image_path == "./assets/steam.png") {
      // handle old default steam image definition
      return SUNSHINE_ASSETS_DIR "/steam.png";
    }

    // check if specified image exists
    std::error_code code;
    if (!std::filesystem::exists(app_image_path, code)) {
      // return default box image if image does not exist
      BOOST_LOG(warning) << "Couldn't find app image at path ["sv << app_image_path << ']';
      return DEFAULT_APP_IMAGE_PATH;
    }

    // image is a png, and not in assets directory
    // return only "content-type" http header compatible image type
    return app_image_path;
  }

  std::optional<std::string> calculate_sha256(const std::string &filename) {
    crypto::md_ctx_t ctx {EVP_MD_CTX_create()};
    if (!ctx) {
      return std::nullopt;
    }

    if (!EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr)) {
      return std::nullopt;
    }

    // Read file and update calculated SHA
    char buf[1024 * 16];
    std::ifstream file(filename, std::ifstream::binary);
    while (file.good()) {
      file.read(buf, sizeof(buf));
      if (!EVP_DigestUpdate(ctx.get(), buf, file.gcount())) {
        return std::nullopt;
      }
    }
    file.close();

    unsigned char result[SHA256_DIGEST_LENGTH];
    if (!EVP_DigestFinal_ex(ctx.get(), result, nullptr)) {
      return std::nullopt;
    }

    // Transform byte-array to string
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (const auto &byte : result) {
      ss << std::setw(2) << (int) byte;
    }
    return ss.str();
  }

  uint32_t calculate_crc32(const std::string &input) {
    boost::crc_32_type result;
    result.process_bytes(input.data(), input.length());
    return result.checksum();
  }

  std::tuple<std::string, std::string> calculate_app_id(const std::string &app_name, std::string app_image_path, int index) {
    // Generate id by hashing name with image data if present
    std::vector<std::string> to_hash;
    to_hash.push_back(app_name);
    auto file_path = validate_app_image_path(app_image_path);
    if (file_path != DEFAULT_APP_IMAGE_PATH) {
      auto file_hash = calculate_sha256(file_path);
      if (file_hash) {
        to_hash.push_back(file_hash.value());
      } else {
        // Fallback to just hashing image path
        to_hash.push_back(file_path);
      }
    }

    // Create combined strings for hash
    std::stringstream ss;
    for_each(to_hash.begin(), to_hash.end(), [&ss](const std::string &s) {
      ss << s;
    });
    auto input_no_index = ss.str();
    ss << index;
    auto input_with_index = ss.str();

    // CRC32 then truncate to signed 32-bit range due to client limitations
    auto id_no_index = std::to_string(abs((int32_t) calculate_crc32(input_no_index)));
    auto id_with_index = std::to_string(abs((int32_t) calculate_crc32(input_with_index)));

    return std::make_tuple(id_no_index, id_with_index);
  }

  /**
   * @brief Migrate the applications stored in the file tree by merging in a new app.
   *
   * This function updates the application entries in *fileTree_p* using the data in *inputTree_p*.
   * If an app with the same UUID as the new app is found, it is replaced.
   * Additionally, empty keys (such as "prep-cmd" or "detached") and keys no longer needed ("launching", "index")
   * are removed from the input.
   *
   * @param fileTree_p Pointer to the JSON object representing the file tree.
   * @param inputTree_p Pointer to the JSON object representing the new app.
   */
  void migrate_apps(nlohmann::json *fileTree_p, nlohmann::json *inputTree_p) {
    std::string new_app_uuid;

    if (inputTree_p) {
      // If the input contains a non-empty "uuid", use it; otherwise generate one.
      if (inputTree_p->contains("uuid") && !(*inputTree_p)["uuid"].get<std::string>().empty()) {
        new_app_uuid = (*inputTree_p)["uuid"].get<std::string>();
      } else {
        new_app_uuid = uuid_util::uuid_t::generate().string();
        (*inputTree_p)["uuid"] = new_app_uuid;
      }

      // Remove "prep-cmd" if empty.
      if (inputTree_p->contains("prep-cmd") && (*inputTree_p)["prep-cmd"].empty()) {
        inputTree_p->erase("prep-cmd");
      }

      // Remove "detached" if empty.
      if (inputTree_p->contains("detached") && (*inputTree_p)["detached"].empty()) {
        inputTree_p->erase("detached");
      }

      // Remove keys that are no longer needed.
      inputTree_p->erase("launching");
      inputTree_p->erase("index");
    }

    // Get the current apps array; if it doesn't exist, create one.
    nlohmann::json newApps = nlohmann::json::array();
    if (fileTree_p->contains("apps") && (*fileTree_p)["apps"].is_array()) {
      for (auto &app : (*fileTree_p)["apps"]) {
        // If an app with the same UUID as the new app is found, replace it.
        if (!new_app_uuid.empty() && app.at("uuid").get<std::string>() == new_app_uuid) {
          newApps.push_back(*inputTree_p);
          new_app_uuid.clear();
        } else {
          newApps.push_back(std::move(app));
        }
      }
    }
    // If the new app's UUID has not been merged yet, add it.
    if (!new_app_uuid.empty() && inputTree_p) {
      newApps.push_back(*inputTree_p);
    }
    (*fileTree_p)["apps"] = newApps;
  }

  std::optional<proc::proc_t> parse(const std::string &file_name) {
    // Prepare environment variables.
    auto this_env = boost::this_process::environment();

    std::set<std::string> ids;
    std::vector<proc::ctx_t> apps;
    int i = 0;

    size_t fail_count = 0;
    do {
      // Read the JSON file into a tree.
      nlohmann::json tree;
      try {
        std::string content = file_handler::read_file(file_name.c_str());
        tree = nlohmann::json::parse(content);
      } catch (const std::exception &e) {
        BOOST_LOG(warning) << "Couldn't read apps.json properly! Apps will not be loaded."sv;
        fail_count = 1;
        break;
      }

      try {
        if (tree.contains("env") && tree["env"].is_object()) {
          for (auto &item : tree["env"].items()) {
            this_env[item.key()] = parse_env_val(this_env, item.value().get<std::string>());
          }
        }

        // Ensure the "apps" array exists.
        if (!tree.contains("apps") || !tree["apps"].is_array()) {
          BOOST_LOG(warning) << "No apps were defined in apps.json!!!"sv;
          break;
        }

        // Iterate over each application in the "apps" array.
        for (auto &app_node : tree["apps"]) {
          proc::ctx_t ctx;
          ctx.idx = std::to_string(i);
          ctx.uuid = app_node.at("uuid");

          // Build the list of preparation commands.
          std::vector<proc::cmd_t> prep_cmds;
          bool exclude_global_prep = app_node.value("exclude-global-prep-cmd", false);
          if (!exclude_global_prep) {
            prep_cmds.reserve(config::sunshine.prep_cmds.size());
            for (auto &prep_cmd : config::sunshine.prep_cmds) {
              auto do_cmd = parse_env_val(this_env, prep_cmd.do_cmd);
              auto undo_cmd = parse_env_val(this_env, prep_cmd.undo_cmd);
              prep_cmds.emplace_back(
                std::move(do_cmd),
                std::move(undo_cmd),
                std::move(prep_cmd.elevated)
              );
            }
          }
          if (app_node.contains("prep-cmd") && app_node["prep-cmd"].is_array()) {
            for (auto &prep_node : app_node["prep-cmd"]) {
              std::string do_cmd = parse_env_val(this_env, prep_node.value("do", ""));
              std::string undo_cmd = parse_env_val(this_env, prep_node.value("undo", ""));
              bool elevated = prep_node.value("elevated", false);
              prep_cmds.emplace_back(
                std::move(do_cmd),
                std::move(undo_cmd),
                std::move(elevated)
              );
            }
          }

          // Build the list of detached commands.
          std::vector<std::string> detached;
          if (app_node.contains("detached") && app_node["detached"].is_array()) {
            for (auto &detached_val : app_node["detached"]) {
              detached.emplace_back(parse_env_val(this_env, detached_val.get<std::string>()));
            }
          }

          // Process other fields.
          if (app_node.contains("output")) {
            ctx.output = parse_env_val(this_env, app_node.value("output", ""));
          }
          std::string name = parse_env_val(this_env, app_node.value("name", ""));
          if (app_node.contains("cmd")) {
            ctx.cmd = parse_env_val(this_env, app_node.value("cmd", ""));
          }
          if (app_node.contains("working-dir")) {
            ctx.working_dir = parse_env_val(this_env, app_node.value("working-dir", ""));
#ifdef _WIN32
            // The working directory, unlike the command itself, should not be quoted.
            boost::erase_all(ctx.working_dir, "\"");
            ctx.working_dir += '\\';
#endif
          }
          if (app_node.contains("image-path")) {
            ctx.image_path = parse_env_val(this_env, app_node.value("image-path", ""));
          }

          ctx.elevated = app_node.value("elevated", false);
          ctx.auto_detach = app_node.value("auto-detach", true);
          ctx.wait_all = app_node.value("wait-all", true);
          ctx.exit_timeout = std::chrono::seconds {app_node.value("exit-timeout", 5)};
          // Calculate a unique application id.
          auto possible_ids = calculate_app_id(name, ctx.image_path, i++);
          if (ids.count(std::get<0>(possible_ids)) == 0) {
            ctx.id = std::get<0>(possible_ids);
          } else {
            ctx.id = std::get<1>(possible_ids);
          }
          ids.insert(ctx.id);

          ctx.name = std::move(name);
          ctx.prep_cmds = std::move(prep_cmds);
          ctx.detached = std::move(detached);

          apps.emplace_back(std::move(ctx));
        }

      } catch (std::exception &e) {
        BOOST_LOG(error) << "Invalid apps.json v2 configuration: "sv << e.what();
        fail_count = 1;
      }

    } while (false);

    if (fail_count > 0) {
      BOOST_LOG(warning) << "No applications configured, adding fallback Desktop entry.";
      proc::ctx_t ctx;
      ctx.idx = std::to_string(i);
      ctx.uuid = FALLBACK_DESKTOP_UUID;  // Placeholder UUID
      ctx.name = "Desktop (fallback)";
      ctx.image_path = parse_env_val(this_env, "desktop-alt.png");
      ctx.elevated = false;
      ctx.auto_detach = true;
      ctx.wait_all = false;  // Desktop doesn't have a specific command to wait for
      ctx.exit_timeout = 5s;

      // Calculate unique ID
      auto possible_ids = calculate_app_id(ctx.name, ctx.image_path, i++);
      if (ids.count(std::get<0>(possible_ids)) == 0) {
        // Avoid using index to generate id if possible
        ctx.id = std::get<0>(possible_ids);
      } else {
        // Fallback to include index on collision
        ctx.id = std::get<1>(possible_ids);
      }
      ids.insert(ctx.id);

      apps.emplace_back(std::move(ctx));
    }

    // Keep a command-free Virtual Display launch target available whenever SudoVDA is ready.
    // It is generated by Apollo rather than stored in apps.json, so a fresh installation and an
    // existing user configuration expose the same private-desktop entry. The dedicated internal
    // flag keeps this behavior without restoring the retired general per-app virtual-display key.
#ifdef _WIN32
    if (vDisplayDriverStatus == VDISPLAY::DRIVER_STATUS::OK) {
      proc::ctx_t ctx;
      ctx.idx = std::to_string(i);
      ctx.uuid = VIRTUAL_DISPLAY_UUID;
      ctx.name = "Virtual Display";
      ctx.image_path = parse_env_val(this_env, "virtual_desktop.png");
      ctx.elevated = false;
      ctx.auto_detach = true;
      ctx.wait_all = false;
      ctx.synthetic_virtual_display = true;
      ctx.exit_timeout = 5s;

      const auto possible_ids = calculate_app_id(ctx.name, ctx.image_path, i++);
      ctx.id = ids.contains(std::get<0>(possible_ids)) ?
                 std::get<1>(possible_ids) :
                 std::get<0>(possible_ids);
      ids.insert(ctx.id);
      apps.emplace_back(std::move(ctx));
    }
#endif

    return proc::proc_t {
      std::move(this_env),
      std::move(apps)
    };
  }

  void refresh(const std::string &file_name, bool needs_terminate) {
    std::lock_guard lock(process_state_mutex);
    if (needs_terminate) {
      proc.terminate(false, false);
    }

#ifdef _WIN32
    size_t fail_count = 0;
    while (fail_count < 5 && vDisplayDriverStatus != VDISPLAY::DRIVER_STATUS::OK) {
      initVDisplayDriver();
      if (vDisplayDriverStatus == VDISPLAY::DRIVER_STATUS::OK) {
        break;
      }

      fail_count += 1;
      std::this_thread::sleep_for(1s);
    }
#endif

    auto proc_opt = proc::parse(file_name);

    if (proc_opt) {
      proc = std::move(*proc_opt);
    }
  }
}  // namespace proc

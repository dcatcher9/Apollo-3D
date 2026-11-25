/**
 * @file src/system_tray.cpp
 * @brief Definitions for the system tray icon and notification system.
 */
// macros
#if defined SUNSHINE_TRAY && SUNSHINE_TRAY >= 1

  #if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #include <accctrl.h>
    #include <aclapi.h>
    #define TRAY_ICON WEB_DIR "images/sunshine3d.ico"
    #define TRAY_ICON_PLAYING WEB_DIR "images/sunshine3d-playing.ico"
    #define TRAY_ICON_LOCKED WEB_DIR "images/sunshine3d-locked.ico"
  #endif

  #define TRAY_MSG_NO_APP_RUNNING "Reload Apps"

  #ifndef BOOST_PROCESS_VERSION
    #define BOOST_PROCESS_VERSION 1
  #endif

  // standard includes
  #include <atomic>
  #include <chrono>
  #include <future>
  #include <memory>
  #include <mutex>
  #include <string>
  #include <thread>

  // lib includes
  #include <tray/src/tray.h>

  // local includes
  #include "config.h"
  #include "confighttp.h"
  #include "logging.h"
  #include "network.h"
  #include "nvhttp.h"
  #include "platform/common.h"
  #include "src/entry_handler.h"
  #ifdef _WIN32
    #include "platform/windows/ar_glasses.h"
  #endif

using namespace std::literals;

// system_tray namespace
namespace system_tray {
  constexpr char product_display_name[] = PROJECT_DISPLAY_NAME;

  static std::atomic tray_initialized = false;

  // Threading variables for all platforms
  static std::thread tray_thread;
  static std::atomic tray_thread_should_exit = false;
  // The tray thread's Windows identity, for reaching its windows when the tray is closed.
  static std::atomic<unsigned long> tray_thread_id = 0;
  static std::mutex tray_lifecycle_mutex;

  void tray_open_ui_cb([[maybe_unused]] struct tray_menu *item) {
    BOOST_LOG(info) << "Opening UI from system tray"sv;
    launch_ui();
  }

  // Ending the session waits for the launch and process locks, and their holders update the
  // tray with a message this thread must answer. Waiting for them here would stop both: a
  // "Force close" during a launch hung the host. The tray thread only starts the work.
  static void terminate_active_session_off_tray_thread() {
    std::thread([]() {
      nvhttp::terminate_active_session();
    }).detach();
  }

  void
    tray_force_stop_cb(struct tray_menu *item) {
    BOOST_LOG(info) << "Force stop from system tray"sv;
    terminate_active_session_off_tray_thread();
  }

  void tray_restart_cb([[maybe_unused]] struct tray_menu *item) {
    BOOST_LOG(info) << "Restarting from system tray"sv;

    platf::restart();
  }

  void tray_quit_cb([[maybe_unused]] struct tray_menu *item) {
    BOOST_LOG(info) << "Quitting from system tray"sv;

  #ifdef _WIN32
    // If we're running in a service, return a special status to
    // tell it to terminate too, otherwise it will just respawn us.
    if (GetConsoleWindow() == nullptr) {
      lifetime::exit_sunshine(ERROR_SHUTDOWN_IN_PROGRESS, true);
      return;
    }
  #endif

    lifetime::exit_sunshine(0, true);
  }

  #ifdef _WIN32
  void tray_monitor_toggle_cb([[maybe_unused]] struct tray_menu *item) {
    if (!ar_glasses::request_local_session(!ar_glasses::local_session_requested())) {
      launch_ui("/config#local-display");
    }
  }

  void tray_monitor_stop_cb([[maybe_unused]] struct tray_menu *item) {
    ar_glasses::request_local_session(false);
  }
  #endif

  // Tray menu
  static struct tray tray = {
    .icon = TRAY_ICON,
    .tooltip = product_display_name,
    .menu =
      (struct tray_menu[]) {
        // todo - use boost/locale to translate menu strings
        {.text = "Open Sunshine 3D", .cb = tray_open_ui_cb},
        {.text = "-"},
        // { .text = "-" },
        // { .text = "Donate",
        //   .submenu =
        //   (struct tray_menu[]) {
        //   { .text = "GitHub Sponsors", .cb = tray_donate_github_cb },
        //   { .text = "MEE6", .cb = tray_donate_mee6_cb },
        //   { .text = "Patreon", .cb = tray_donate_patreon_cb },
        //   { .text = "PayPal", .cb = tray_donate_paypal_cb },
        //   { .text = nullptr } } },
        // { .text = "-" },
        {.text = TRAY_MSG_NO_APP_RUNNING, .cb = tray_force_stop_cb},
        #ifdef _WIN32
        {.text = "Toggle local 3D", .cb = tray_monitor_toggle_cb},
        {.text = "Return local display to 2D", .cb = tray_monitor_stop_cb},
        #endif
        {.text = "Restart", .cb = tray_restart_cb},
        {.text = "Quit", .cb = tray_quit_cb},
        {.text = nullptr}
      },
    .iconPathCount = 3,
    .allIconPaths = {TRAY_ICON, TRAY_ICON_LOCKED, TRAY_ICON_PLAYING},
  };

  int init_tray() {
  #ifdef _WIN32
    // If we're running as SYSTEM, Explorer.exe will not have permission to open our thread handle
    // to monitor for thread termination. If Explorer fails to open our thread, our tray icon
    // will persist forever if we terminate unexpectedly. To avoid this, we will modify our thread
    // DACL to add an ACE that allows SYNCHRONIZE access to Everyone.
    {
      PACL old_dacl;
      PSECURITY_DESCRIPTOR sd;
      auto error = GetSecurityInfo(GetCurrentThread(), SE_KERNEL_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, &old_dacl, nullptr, &sd);
      if (error != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "GetSecurityInfo() failed: "sv << error;
        return 1;
      }

      auto free_sd = util::fail_guard([sd]() {
        LocalFree(sd);
      });

      SID_IDENTIFIER_AUTHORITY sid_authority = SECURITY_WORLD_SID_AUTHORITY;
      PSID world_sid;
      if (!AllocateAndInitializeSid(&sid_authority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, &world_sid)) {
        error = GetLastError();
        BOOST_LOG(warning) << "AllocateAndInitializeSid() failed: "sv << error;
        return 1;
      }

      auto free_sid = util::fail_guard([world_sid]() {
        FreeSid(world_sid);
      });

      EXPLICIT_ACCESS ea {};
      ea.grfAccessPermissions = SYNCHRONIZE;
      ea.grfAccessMode = GRANT_ACCESS;
      ea.grfInheritance = NO_INHERITANCE;
      ea.Trustee.TrusteeForm = TRUSTEE_IS_SID;
      ea.Trustee.ptstrName = (LPSTR) world_sid;

      PACL new_dacl;
      error = SetEntriesInAcl(1, &ea, old_dacl, &new_dacl);
      if (error != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "SetEntriesInAcl() failed: "sv << error;
        return 1;
      }

      auto free_new_dacl = util::fail_guard([new_dacl]() {
        LocalFree(new_dacl);
      });

      error = SetSecurityInfo(GetCurrentThread(), SE_KERNEL_OBJECT, DACL_SECURITY_INFORMATION, nullptr, nullptr, new_dacl, nullptr);
      if (error != ERROR_SUCCESS) {
        BOOST_LOG(warning) << "SetSecurityInfo() failed: "sv << error;
        return 1;
      }
    }

    // Wait for the shell to be initialized before registering the tray icon.
    // This ensures the tray icon works reliably after a logoff/logon cycle.
    while (GetShellWindow() == nullptr) {
      if (tray_thread_should_exit) {
        return 1;
      }
      Sleep(250);
    }
  #endif

    {
      auto lock = std::lock_guard(tray_lifecycle_mutex);
      if (tray_thread_should_exit) {
        return 1;
      }
      if (tray_init(&tray) < 0) {
        BOOST_LOG(warning) << "Failed to create system tray"sv;
        return 1;
      }

      BOOST_LOG(info) << "System tray created"sv;
      tray_initialized = true;
    }
    return 0;
  }

  int process_tray_events() {
    if (!tray_initialized) {
      return 1;
    }

  #ifdef _WIN32
    // The Windows tray owns a dedicated thread, so block until an event or tray_exit().
    constexpr int blocking = 1;
  #else
    // Other platforms share the main loop and must remain non-blocking.
    constexpr int blocking = 0;
  #endif
    if (const int result = tray_loop(blocking); result != 0) {
      if (!tray_thread_should_exit) {
        BOOST_LOG(warning) << "System tray loop failed"sv;
      }
      return result;
    }

    return 0;
  }

  int end_tray() {
    bool owned_tray = false;
    {
      auto lock = std::lock_guard(tray_lifecycle_mutex);
      owned_tray = tray_initialized.exchange(false);
    }
    if (owned_tray) {
      tray_exit();
    }
    return 0;
  }

  void update_tray_playing(std::string app_name) {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return;
    }

    tray.notification_title = nullptr;
    tray.notification_text = nullptr;
    tray.notification_cb = nullptr;
    tray.notification_icon = nullptr;
    tray.icon = TRAY_ICON_PLAYING;

    tray_update(&tray);
    tray.icon = TRAY_ICON_PLAYING;
    tray.notification_title = "App launched";
    char msg[256];
    static char force_close_msg[256];
    snprintf(msg, std::size(msg), "%s launched.", app_name.c_str());
    snprintf(force_close_msg, std::size(force_close_msg), "Force close [%s]", app_name.c_str());
    tray.notification_text = msg;
    tray.notification_icon = TRAY_ICON_PLAYING;
    tray.tooltip = product_display_name;
    tray.menu[2].text = force_close_msg;
    tray_update(&tray);
  }

  void update_tray_stopped(std::string app_name) {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return;
    }

    tray.notification_title = nullptr;
    tray.notification_text = nullptr;
    tray.notification_cb = nullptr;
    tray.notification_icon = nullptr;
    tray.icon = TRAY_ICON;
    tray_update(&tray);
    char msg[256];
    snprintf(msg, std::size(msg), "Streaming stopped for %s", app_name.c_str());
    tray.icon = TRAY_ICON;
    tray.notification_icon = TRAY_ICON;
    tray.notification_title = "Application Stopped";
    tray.notification_text = msg;
    tray.tooltip = product_display_name;
    tray.menu[2].text = TRAY_MSG_NO_APP_RUNNING;
    tray_update(&tray);
  }

  void
    update_tray_launch_error(std::string app_name, int exit_code) {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return;
    }

    tray.notification_title = NULL;
    tray.notification_text = NULL;
    tray.notification_cb = NULL;
    tray.notification_icon = NULL;
    tray.icon = TRAY_ICON;
    tray_update(&tray);
    char msg[256];
    snprintf(msg, std::size(msg), "Application %s exited too fast with code %d. Click here to terminate the stream.", app_name.c_str(), exit_code);
    tray.icon = TRAY_ICON;
    tray.notification_icon = TRAY_ICON;
    tray.notification_title = "Launch Error";
    tray.notification_text = msg;
    tray.notification_cb = []() {
      BOOST_LOG(info) << "Force stop from notification"sv;
      terminate_active_session_off_tray_thread();
    };
    tray.tooltip = product_display_name;
    tray_update(&tray);
  }

  void update_tray_require_pin() {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return;
    }

    tray.notification_title = nullptr;
    tray.notification_text = nullptr;
    tray.notification_cb = nullptr;
    tray.notification_icon = nullptr;
    tray.icon = TRAY_ICON;
    tray_update(&tray);
    tray.icon = TRAY_ICON;
    tray.notification_title = "Incoming Pairing Request";
    tray.notification_text = "Click here to complete the pairing process";
    tray.notification_icon = TRAY_ICON_LOCKED;
    tray.tooltip = product_display_name;
    tray.notification_cb = []() {
      launch_ui("/pin#PIN");
    };
    tray_update(&tray);
  }

  bool update_tray_ar_display_decision(std::string display_name) {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return false;
    }

    tray.notification_title = nullptr;
    tray.notification_text = nullptr;
    tray.notification_cb = nullptr;
    tray.notification_icon = nullptr;
    tray_update(&tray);
    char msg[256];
    snprintf(
      msg,
      std::size(msg),
      "Is '%s' an AR display? Click to choose in Sunshine 3D.",
      display_name.c_str()
    );
    tray.notification_title = "New Monitor Detected";
    tray.notification_text = msg;
    tray.notification_icon = TRAY_ICON;
    tray.tooltip = product_display_name;
    tray.notification_cb = []() {
      launch_ui("/config#local-display");
    };
    tray_update(&tray);
    return true;
  }

  bool update_tray_game3d_setup(std::string game_name, std::string detail) {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return false;
    }

    tray.notification_title = nullptr;
    tray.notification_text = nullptr;
    tray.notification_cb = nullptr;
    tray.notification_icon = nullptr;
    tray_update(&tray);
    char msg[256];
    snprintf(msg, std::size(msg), "%s: %s Click to view setup.", game_name.c_str(), detail.c_str());
    tray.notification_title = "Game 3D Setup";
    tray.notification_text = msg;
    tray.notification_icon = TRAY_ICON;
    tray.tooltip = product_display_name;
    tray.notification_cb = []() {
      launch_ui("/#game3d-setup");
    };
    tray_update(&tray);
    return true;
  }

  void
    update_tray_paired(std::string device_name) {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return;
    }

    tray.notification_title = NULL;
    tray.notification_text = NULL;
    tray.notification_cb = NULL;
    tray.notification_icon = NULL;
    tray_update(&tray);
    char msg[256];
    snprintf(msg, std::size(msg), "Device %s paired Succesfully. Please make sure you have access to the device.", device_name.c_str());
    tray.notification_title = "Device Paired Succesfully";
    tray.notification_text = msg;
    tray.notification_icon = TRAY_ICON;
    tray.tooltip = product_display_name;
    tray_update(&tray);
  }

  void
    update_tray_client_connected(std::string client_name) {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return;
    }

    tray.notification_title = NULL;
    tray.notification_text = NULL;
    tray.notification_cb = NULL;
    tray.notification_icon = NULL;
    tray.icon = TRAY_ICON;
    tray_update(&tray);
    char msg[256];
    snprintf(msg, std::size(msg), "%s has connected to the session.", client_name.c_str());
    tray.notification_title = "Client Connected";
    tray.notification_text = msg;
    tray.notification_icon = TRAY_ICON;
    tray.tooltip = product_display_name;
    tray_update(&tray);
  }

  bool update_tray_notice(std::string title, std::string text) {
    auto lock = std::lock_guard(tray_lifecycle_mutex);
    if (!tray_initialized) {
      return false;
    }

    tray.notification_title = nullptr;
    tray.notification_text = nullptr;
    tray.notification_cb = nullptr;
    tray.notification_icon = nullptr;
    tray_update(&tray);
    tray.notification_title = title.c_str();
    tray.notification_text = text.c_str();
    tray.notification_icon = TRAY_ICON;
    tray.tooltip = product_display_name;
    tray.notification_cb = []() {
      launch_ui("/troubleshooting");
    };
    tray_update(&tray);
    // The strings above end with this call: nothing may read them afterwards.
    tray.notification_title = nullptr;
    tray.notification_text = nullptr;
    return true;
  }

  // Threading functions available on all platforms
  static void tray_thread_worker() {
    BOOST_LOG(info) << "System tray thread started"sv;
  #ifdef _WIN32
    tray_thread_id = GetCurrentThreadId();
  #endif

    // Initialize the tray in this thread
    if (init_tray() != 0) {
      if (!tray_thread_should_exit) {
        BOOST_LOG(error) << "Failed to initialize tray in thread"sv;
      }
      return;
    }

    // Block for tray events until end_tray() wakes the loop. Once the tray window is gone the
    // loop call returns at once without a quit message, so the exit flag ends the loop too.
    while (!tray_thread_should_exit && process_tray_events() == 0);

    BOOST_LOG(info) << "System tray thread ended"sv;
  }

  int init_tray_threaded() {
    if (tray_thread.joinable()) {
      BOOST_LOG(warning) << "Tray thread is already running"sv;
      return 1;
    }

    static const std::string title_str = "Open Sunshine 3D (" + config::nvhttp.sunshine_name + ":" + std::to_string(net::map_port(confighttp::PORT_HTTPS)) + ")";
    tray.menu[0].text = title_str.c_str();

    if (config::sunshine.hide_tray_controls) {
      tray.menu[1].text = nullptr;
    }

    tray_thread_should_exit = false;

    try {
      tray_thread = std::thread(tray_thread_worker);
      BOOST_LOG(info) << "System tray thread launched"sv;
      return 0;
    } catch (const std::exception &e) {
      BOOST_LOG(error) << "Failed to create tray thread: " << e.what();
      return 1;
    }
  }

  int end_tray_threaded() {
    if (!tray_thread.joinable()) {
      return 0;
    }

    BOOST_LOG(info) << "Stopping system tray thread"sv;
    tray_thread_should_exit = true;

  #ifdef _WIN32
    // A tray menu that is still open keeps the tray thread inside the menu's own message loop.
    // Close it first, by asking each of that thread's windows to leave any modal state.
    EnumThreadWindows(
      tray_thread_id.load(),
      [](HWND window, LPARAM) -> BOOL {
        PostMessageW(window, WM_CANCELMODE, 0, 0);
        return TRUE;
      },
      0
    );
  #endif

    // Closing the tray sends a message to its thread and waits for the answer, and the thread
    // itself may be inside a menu or a callback. The host is exiting or restarting, so neither
    // may hold it: wait a bounded time for each, then leave the thread behind.
    constexpr auto patience = 2s;
    auto closed = std::make_shared<std::promise<void>>();
    auto closed_future = closed->get_future();
    std::thread closer([closed]() {
      end_tray();
      closed->set_value();
    });
    if (closed_future.wait_for(patience) != std::future_status::ready) {
      BOOST_LOG(warning) << "System tray did not close in time; continuing without it"sv;
      closer.detach();
      tray_thread.detach();
      return 0;
    }
    closer.join();

    auto ended = std::make_shared<std::promise<void>>();
    auto ended_future = ended->get_future();
    // The joiner owns the thread from here: left behind on a timeout, it must not leave the
    // static handle joinable, whose destruction at exit would end the process.
    std::thread joiner([ended, thread = std::move(tray_thread)]() mutable {
      thread.join();
      ended->set_value();
    });
    if (ended_future.wait_for(patience) != std::future_status::ready) {
      BOOST_LOG(warning) << "System tray thread did not end in time; continuing without it"sv;
      joiner.detach();
      return 0;
    }
    joiner.join();

    BOOST_LOG(info) << "System tray thread stopped"sv;
    return 0;
  }

}  // namespace system_tray

  #ifdef BOOST_PROCESS_VERSION
    #undef BOOST_PROCESS_VERSION
  #endif

#endif

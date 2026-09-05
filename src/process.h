/**
 * @file src/process.h
 * @brief Declarations for the startup and shutdown of the apps started by a streaming Session.
 */
#pragma once

#ifndef __kernel_entry
  #define __kernel_entry
#endif

#ifndef BOOST_PROCESS_VERSION
  #define BOOST_PROCESS_VERSION 1
#endif

// standard includes
#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stop_token>
#include <thread>
#include <unordered_map>

// lib includes
#include <boost/process/v1/child.hpp>
#include <boost/process/v1/environment.hpp>
#include <boost/process/v1/group.hpp>
#include <boost/process/v1/search_path.hpp>
#include <boost/property_tree/ptree.hpp>
#include <nlohmann/json.hpp>

// local includes
#include "config.h"
#include "platform/common.h"
#include "rtsp.h"
#include "utility.h"

#ifdef _WIN32
  #include "platform/windows/virtual_display.h"
#endif

#define VIRTUAL_DISPLAY_UUID "8902CB19-674A-403D-A587-41B092E900BA"
#define FALLBACK_DESKTOP_UUID "EAAC6159-089A-46A9-9E24-6436885F6610"

namespace proc {
  using file_t = util::safe_ptr_v2<FILE, int, fclose>;

#ifdef _WIN32
  extern std::atomic<VDISPLAY::DRIVER_STATUS> vDisplayDriverStatus;

  struct hdr_worker_state_t;

  enum class local_ar_handoff_e {
    ready,
    remote_busy,
    cleanup_timeout,
  };

  #ifdef SUNSHINE_TESTS
  bool retiredVirtualDisplayRemovalAllowedForTest(
    bool desktopDeactivationRequested,
    bool detachConfirmedReady
  );
  bool virtualDisplayRetirementHandoffMarksSessionForTest(
    bool hasBoundDisplay,
    bool retiringBoundDisplay
  );
  bool explorerRepairAllowedForRetirementForTest(
    bool finalTeardown,
    bool restartEnabled,
    bool activeDesktopRemovalDebt
  );
  #endif
#endif

  typedef config::prep_cmd_t cmd_t;

  /**
   * pre_cmds -- guaranteed to be executed unless any of the commands fail.
   * detached -- commands detached from Sunshine
   * cmd -- Runs indefinitely until:
   *    No session is running and a different set of commands it to be executed
   *    Command exits
   * working_dir -- the process working directory. This is required for some games to run properly.
   * cmd_output --
   *    empty    -- The output of the commands are appended to the output of sunshine
   *    "null"   -- The output of the commands are discarded
   *    filename -- The output of the commands are appended to filename
   */
  struct ctx_t {
    std::vector<cmd_t> prep_cmds;

    /**
     * Some applications, such as Steam, either exit quickly, or keep running indefinitely.
     *
     * Apps that launch normal child processes and terminate will be handled by the process
     * grouping logic (wait_all). However, apps that launch child processes indirectly or
     * into another process group (such as UWP apps) can only be handled by the auto-detach
     * heuristic which catches processes that exit 0 very quickly, but we won't have proper
     * process tracking for those.
     *
     * For cases where users just want to kick off a background process and never manage the
     * lifetime of that process, they can use detached commands for that.
     */
    std::vector<std::string> detached;

    std::string idx;
    std::string uuid;
    std::string name;
    std::string cmd;
    std::string working_dir;
    std::string output;
    std::string image_path;
    std::string id;
    bool elevated;
    bool auto_detach;
    bool wait_all;
    // Set only on Apollo's generated Virtual Display tile. This deliberately does not restore
    // the retired user-configurable per-app virtual-display switch.
    bool synthetic_virtual_display = false;
    std::chrono::seconds exit_timeout;
  };

  struct process_status_t {
    int app_id;
    std::string app_name;
    std::string app_uuid;
    bool virtual_display;
    std::uint64_t host_session_id;
    bool enable_hdr;
    std::string client_uuid;
  };

  struct render_size_t {
    std::uint32_t width;
    std::uint32_t height;

    bool operator==(const render_size_t &) const = default;
  };

  /** Scale a client mode exactly as the production render path does. */
  std::optional<render_size_t> calculate_render_size(
    std::uint32_t client_width,
    std::uint32_t client_height,
    std::uint32_t scale_factor
  );

  /**
   * Outcome of a live (no-reconnect) stream geometry/rate change.
   * The underlying type is fixed so stream.h can declare this opaquely for its 0x3008
   * acknowledgement mapping without including this header.
   */
  enum class live_video_mode_result_e : int {
    applied,  ///< The desktop now presents the requested mode.
    unchanged,  ///< Nothing had to change; the encoder may rebuild immediately.
    needs_reconnect,  ///< A fresh launch is required; the live desktop may no longer be proven.
    failed,  ///< Attempted and rolled back; the session keeps its previous mode.
  };

  /** A failed apply is retryable only when the previous desktop contract was proven restored. */
  [[nodiscard]] constexpr live_video_mode_result_e live_video_mode_failure_result(
    bool rollback_succeeded
  ) noexcept {
    return rollback_succeeded ?
             live_video_mode_result_e::failed :
             live_video_mode_result_e::needs_reconnect;
  }

  class proc_t {
  public:
    KITTY_DEFAULT_CONSTR_MOVE_THROW(proc_t)

    std::string initial_display;

    proc_t(
      boost::process::v1::environment &&env,
      std::vector<ctx_t> &&apps
    ):
        _env(std::move(env)),
        _apps(std::move(apps)) {
    }

    int execute(
      const ctx_t &app,
      std::shared_ptr<rtsp_stream::launch_session_t> launch_session,
      bool probe_encoder
    );

    /**
     * @return `_app_id` if a process is running, otherwise returns `0`
     */
    int running();

    /** True when the command-backed app ended; does not tear down display/process state. */
    bool stream_process_exited();

    /** Coherent process/session state after refreshing the tracked child lifetime. */
    process_status_t get_status();

    ~proc_t();

    std::vector<ctx_t> get_apps() const;
    std::string get_app_image(int app_id);
    std::string get_last_run_app_name();
    std::string get_running_app_uuid();
    /** Stable retained process-instance token. Zero means no resumable process exists. */
    std::uint64_t get_host_session_id() const;
    /** Apply a reconnect's render/display/HDR contract without relaunching the application. */
    int reconfigure_retained_session(std::shared_ptr<rtsp_stream::launch_session_t> launch_session);
    /**
     * Resize the session's virtual desktop in place for a live 0x3007 mode change. The streaming
     * session, its capture pipeline and its RTSP session all survive. Never recreates the monitor:
     * recreation retires it from the Windows topology and would kill the in-flight capture.
     * @param width Requested desktop width in pixels.
     * @param height Requested desktop height in pixels.
     * @param fps_millihz Requested refresh rate in millihertz, matching launch_session_t::fps.
     * @param expected_launch_session_id Session identity captured with the control request. A
     * stale worker must never resize a replacement session's desktop.
     */
    live_video_mode_result_e apply_live_video_mode(
      int width,
      int height,
      int fps_millihz,
      std::uint32_t expected_launch_session_id
    );
    /**
     * Non-blocking hint for the serialized live-mode worker: would `apply_live_video_mode` have
     * real topology work to do? A bitrate-only or frame-rate-only change skips DisplayConfig but
     * remains ordered behind any active encoder transaction. This never waits on the process lock;
     * a transition already in flight conservatively answers "yes".
     */
    bool live_video_mode_needs_display_change(int width, int height) const;
    /** Adopt the remote streaming session's virtual-display lease before platform startup. */
    bool activate_remote_virtual_display_lease(std::uint64_t lease);
    void terminate(bool immediate = false, bool needs_refresh = true);

    /** Thread-safe snapshot/update of the display selected by the capture pipeline. */
    std::string get_display_name() const;
    void set_display_name(std::string name);

#ifdef _WIN32
    /**
     * Release an inactive remote virtual desktop before local AR creates its own source.
     * Active or connecting remote sessions are never terminated.
     */
    local_ar_handoff_e prepare_local_ar_handoff(const std::stop_source &construction_stop);
    /**
     * Give an orderly host shutdown one final bounded chance to finish deferred monitor removal
     * and the optional Explorer repair before the retirement workers are stopped.
     */
    static bool drain_retired_virtual_display_for_shutdown(
      std::chrono::milliseconds timeout
    );
#endif

  private:
    int running_locked();
    bool stream_process_exited_locked();
    void set_display_name_locked(std::string name);

    int _app_id = 0;
    std::string _app_name;
    bool _virtual_display = false;
    std::uint64_t _host_session_id = 0;

    boost::process::v1::environment _env;

    std::shared_ptr<rtsp_stream::launch_session_t> _launch_session;
#ifdef _WIN32
    std::optional<SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT> _virtual_display_identity;
    std::optional<LUID> _virtual_display_render_adapter;
    std::wstring _virtual_display_device_path;
    std::wstring _virtual_display_gdi_name;
    bool _virtual_display_published = false;
    bool _virtual_display_retirement_handed_off = false;
    std::optional<std::uint64_t> _remote_virtual_display_lease;
    VDISPLAY::creation_result_t create_retained_virtual_display(
      std::uint32_t width,
      std::uint32_t height,
      std::uint32_t fps,
      const GUID &guid,
      const std::optional<LUID> &render_adapter
    );
    bool retire_virtual_display(
      const std::optional<SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT> &identity,
      const GUID &guid,
      const std::wstring &device_path,
      const std::wstring &gdi_name,
      bool was_published,
      std::chrono::milliseconds timeout,
      bool deactivate_desktop,
      bool final_teardown,
      bool restart_explorer_after_removal
    );
    static bool prepare_retired_virtual_display_for_removal(std::chrono::milliseconds timeout);
    static void schedule_retired_virtual_display_cleanup();
    static bool has_retired_virtual_display();
    void clear_virtual_display_binding();
    void adopt_virtual_display(
      VDISPLAY::creation_result_t created_display,
      bool enable_hdr
    );
    static bool wait_for_retired_virtual_display(
      std::chrono::milliseconds timeout,
      bool wait_for_explorer_repair = true
    );
    void start_hdr_worker(bool enable_hdr);
    bool request_hdr_state(bool enable_hdr, std::chrono::milliseconds timeout);
    void stop_hdr_worker();

    std::shared_ptr<hdr_worker_state_t> _hdr_worker_state;
    std::jthread _hdr_worker;
#endif

    std::string display_name;

    std::vector<ctx_t> _apps;
    ctx_t _app;
    std::chrono::steady_clock::time_point _app_launch_time;

    // If no command associated with _app_id, yet it's still running
    bool placebo {};

    boost::process::v1::child _process;
    boost::process::v1::group _process_group;

    file_t _pipe;
    std::size_t _completed_prep_commands = 0;
  };

  boost::filesystem::path
    find_working_directory(const std::string &cmd, const boost::process::v1::environment &env);

  /**
   * @brief Calculate a stable id based on name and image data
   * @return Tuple of id calculated without index (for use if no collision) and one with.
   */
  std::tuple<std::string, std::string> calculate_app_id(const std::string &app_name, std::string app_image_path, int index);

  std::string validate_app_image_path(std::string app_image_path);
  void refresh(const std::string &file_name, bool needs_terminate = true);
  void migrate_apps(nlohmann::json *fileTree_p, nlohmann::json *inputTree_p);
  std::optional<proc::proc_t> parse(const std::string &file_name);

  /**
   * @brief Initialize proc functions
   * @return Unique pointer to `deinit_t` to manage cleanup
   */
  std::unique_ptr<platf::deinit_t> init();

  /**
   * @brief Terminates all child processes in a process group.
   * @param proc The child process itself.
   * @param group The group of all children in the process tree.
   * @param exit_timeout The timeout to wait for the process group to gracefully exit.
   */
  void terminate_process_group(boost::process::v1::child &proc, boost::process::v1::group &group, std::chrono::seconds exit_timeout);

  extern proc_t proc;

}  // namespace proc

#ifdef BOOST_PROCESS_VERSION
  #undef BOOST_PROCESS_VERSION
#endif

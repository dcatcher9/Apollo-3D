/**
 * @file src/platform/windows/virtual_display_session.h
 * @brief Explicit ownership of one virtual display across acquisition, pause and retirement.
 */
#pragma once

#include "primary_display.h"
#include "virtual_display.h"

namespace VDISPLAY {
  struct display_spec_t {
    std::string client_uid;
    std::string client_name;
    GUID guid {};
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t refresh_millihz = 0;
    std::optional<LUID> render_adapter_luid;
    std::wstring adapter_name;
    bool exclusive = false;
    std::wstring local_sink;
    // Extended local AR owns its existing row journal and supplies acquire's prepare callback.
    bool primary_binding = true;
  };

  struct session_io_t {
    std::function<bool(const display_spec_t &)> prepare;
    std::function<creation_result_t(const display_spec_t &)> create;
    std::function<bool(std::wstring_view)> bind;
    std::function<bool(std::wstring_view, bool)> promote;
    std::function<bool(std::wstring_view)> restore;
    std::function<bool(std::wstring_view, platf::primary_display::retained_display_ptr &, std::wstring_view)> pause;
    std::function<bool(const platf::primary_display::retained_display_ptr &, bool)> reactivate;
    std::function<display_identity_query_t(const creation_result_t &)> query;
    std::function<display_identity_state_e(const creation_result_t &)> query_retirement;
    std::function<bool(const GUID &)> remove;
    std::function<std::chrono::steady_clock::time_point()> now;
    std::function<void(std::chrono::milliseconds)> sleep;
  };

  class session_t;

  struct session_retirement_callbacks_t {
    // Defaults to restore(). A failed prepare is retried before any driver removal.
    std::function<bool(session_t &)> prepare;
    // Revalidate adapter-specific detach safety immediately before each Remove attempt.
    std::function<bool(session_t &)> before_remove;
    std::function<display_identity_state_e(const session_t &)> query;
    // Successful cleanup is latched before the final absence query and is never replayed.
    std::function<bool(session_t &)> finish;
  };

  struct session_retirement_policy_t {
    retirement_timing_t timing {};
    std::chrono::milliseconds remove_retry_interval {0};
    bool require_device_absence = true;
    bool retry_prepare_within_slice = false;
    bool retry_acknowledged_removal_each_slice = false;
  };

  class session_t {
  public:
    using time_point = std::chrono::steady_clock::time_point;
    session_t();
    // Unspecified operations use the production adapter; fake tests override every I/O operation.
    explicit session_t(session_io_t io);
    ~session_t() = default;  // Never performs I/O; unresolved ownership must be transferred.
    session_t(const session_t &) = delete;
    session_t &operator=(const session_t &) = delete;
    session_t(session_t &&other) noexcept;
    // Transfer requires an empty/completed destination. Reset a moved-from owner before reuse.
    session_t &operator=(session_t &&other);

    bool acquire(display_spec_t spec, const std::function<bool()> &prepare = {}, const std::function<bool()> &cancelled = {});
    // Adopts even an unpublished Add result. Refuses to overwrite an existing owner.
    bool adopt(display_spec_t spec, creation_result_t binding, bool prepared = true);
    const display_spec_t &spec() const;
    const creation_result_t &binding() const;
    bool owns_display() const;
    bool prepared() const;
    bool was_published() const;
    bool pause_requested() const;
    bool paused() const;
    bool has_retained() const;
    bool retiring() const;
    bool removal_accepted() const;
    bool retirement_complete() const;
    std::uint64_t generation() const;
    void set_exclusive(bool exclusive);
    void update_mode(std::uint32_t width, std::uint32_t height, std::uint32_t refresh_millihz);

    // The caller has already verified this binding against the owner's stable identity.
    bool update_binding(creation_result_t binding);
    void invalidate_gdi();
    bool refresh();
    bool bind();
    bool promote();
    bool restore();
    bool pause(std::wstring_view local_sink = {});
    // Reactivates only. The adapter verifies publication using its existing retry window before
    // committing resume, since Windows may publish the new GDI binding asynchronously.
    bool begin_resume();
    void commit_resume();

    // Callbacks are stored by value and must not capture the originating session object.
    void begin_retirement(session_retirement_callbacks_t callbacks = {}, session_retirement_policy_t policy = {});
    bool retire_until(time_point deadline);
    void invalidate_retirement_preparation();

  private:
    void clear_moved_state();
    display_identity_state_e retirement_query() const;
    session_io_t io_;
    display_spec_t spec_;
    creation_result_t binding_;
    bool prepared_ = false;
    bool was_published_ = false;
    bool pause_requested_ = false;
    bool paused_ = false;
    platf::primary_display::retained_display_ptr retained_;
    std::uint64_t generation_ = 0;
    std::optional<retirement_record_t> retirement_;
    session_retirement_callbacks_t retirement_callbacks_;
    session_retirement_policy_t retirement_policy_;
    bool retirement_prepared_ = false;
    bool removal_accepted_ = false;
    bool cleanup_finished_ = false;
    bool retirement_complete_ = false;
    time_point next_remove_attempt_ {};
  };
}  // namespace VDISPLAY

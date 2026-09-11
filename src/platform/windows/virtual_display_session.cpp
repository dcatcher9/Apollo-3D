#include "virtual_display_session.h"

#include <atomic>
#include <new>
#include <stdexcept>
#include <utility>

namespace VDISPLAY {
  namespace {
    std::atomic<std::uint64_t> next_generation {0};

    bool same_identity(const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &left, const SUDOVDA::VIRTUAL_DISPLAY_ADD_OUT &right) {
      return left.AdapterLuid.HighPart == right.AdapterLuid.HighPart &&
             left.AdapterLuid.LowPart == right.AdapterLuid.LowPart && left.TargetId == right.TargetId;
    }

    session_io_t production_io() {
      return {
        .prepare = [](const display_spec_t &spec) {
          return spec.exclusive && !spec.local_sink.empty() ?
                   platf::primary_display::prepare_local_exclusive(spec.local_sink) :
                   platf::primary_display::prepare(spec.exclusive);
        },
        .create = [](const display_spec_t &spec) {
          if (spec.render_adapter_luid) {
            return createVirtualDisplayOnAdapter(spec.client_uid.c_str(), spec.client_name.c_str(), spec.width, spec.height, spec.refresh_millihz, spec.guid, *spec.render_adapter_luid);
          }
          if (!spec.adapter_name.empty()) {
            return createVirtualDisplayOnAdapter(spec.client_uid.c_str(), spec.client_name.c_str(), spec.width, spec.height, spec.refresh_millihz, spec.guid, spec.adapter_name);
          }
          return createVirtualDisplay(spec.client_uid.c_str(), spec.client_name.c_str(), spec.width, spec.height, spec.refresh_millihz, spec.guid);
        },
        .bind = [](std::wstring_view path) { return platf::primary_display::bind_pending(path); },
        .promote = [](std::wstring_view path, bool exclusive) { return platf::primary_display::promote(path, exclusive); },
        .restore = [](std::wstring_view path) { return platf::primary_display::restore(path); },
        .pause = [](std::wstring_view path, auto &retained, std::wstring_view sink) { return platf::primary_display::pause(path, retained, sink); },
        .reactivate = [](const auto &retained, bool exclusive) { return platf::primary_display::reactivate(retained, exclusive); },
        .query = [](const creation_result_t &binding) {
          return binding.identity ? queryVirtualDisplayIdentity(*binding.identity, binding.device_path, binding.display_name) : display_identity_query_t {};
        },
        .query_retirement = [](const creation_result_t &binding) {
          return binding.identity ? queryVirtualDisplayRetirementState(*binding.identity, binding.device_path, binding.display_name) : display_identity_state_e::indeterminate;
        },
        .remove = [](const GUID &guid) { return removeVirtualDisplay(guid); },
        .now = []() { return std::chrono::steady_clock::now(); },
        .sleep = [](std::chrono::milliseconds delay) { std::this_thread::sleep_for(delay); },
        .restore_cursor = [](const auto &retained) { platf::primary_display::restore_retained_cursor(retained); },
      };
    }
  }

  session_t::session_t(): io_(production_io()) {}
  session_t::session_t(session_io_t io): session_t() {
    const auto override_if_set = [](auto &target, auto &override) {
      if (override) {
        target = std::move(override);
      }
    };
    override_if_set(io_.prepare, io.prepare);
    override_if_set(io_.create, io.create);
    override_if_set(io_.bind, io.bind);
    override_if_set(io_.promote, io.promote);
    override_if_set(io_.restore, io.restore);
    override_if_set(io_.pause, io.pause);
    override_if_set(io_.reactivate, io.reactivate);
    override_if_set(io_.query, io.query);
    override_if_set(io_.query_retirement, io.query_retirement);
    override_if_set(io_.remove, io.remove);
    override_if_set(io_.now, io.now);
    override_if_set(io_.sleep, io.sleep);
    override_if_set(io_.restore_cursor, io.restore_cursor);
  }

  session_t::session_t(session_t &&other) noexcept:
      io_(std::move(other.io_)),
      spec_(std::move(other.spec_)),
      binding_(std::move(other.binding_)),
      prepared_(other.prepared_),
      was_published_(other.was_published_),
      pause_requested_(other.pause_requested_),
      paused_(other.paused_),
      retained_(std::move(other.retained_)),
      generation_(other.generation_),
      retirement_(std::move(other.retirement_)),
      retirement_callbacks_(std::move(other.retirement_callbacks_)),
      retirement_policy_(other.retirement_policy_),
      retirement_prepared_(other.retirement_prepared_),
      removal_accepted_(other.removal_accepted_),
      cleanup_finished_(other.cleanup_finished_),
      retirement_complete_(other.retirement_complete_),
      next_remove_attempt_(other.next_remove_attempt_) {
    other.clear_moved_state();
  }

  session_t &session_t::operator=(session_t &&other) {
    if (this == &other) {
      return *this;
    }
    if (owns_display() || prepared_ || retained_) {
      throw std::logic_error("Cannot overwrite unresolved virtual-display ownership");
    }
    // Move construction transfers all proof, including stored adapter cleanup and retry timing.
    this->~session_t();
    new (this) session_t(std::move(other));
    return *this;
  }

  void session_t::clear_moved_state() {
    spec_ = {};
    binding_ = {};
    prepared_ = was_published_ = pause_requested_ = paused_ = false;
    retained_.reset();
    generation_ = 0;
    retirement_.reset();
    retirement_callbacks_ = {};
    retirement_policy_ = {};
    retirement_prepared_ = removal_accepted_ = cleanup_finished_ = retirement_complete_ = false;
    next_remove_attempt_ = {};
  }

  bool session_t::acquire(display_spec_t spec, const std::function<bool()> &prepare, const std::function<bool()> &cancelled) {
    if (owns_display() || prepared_ || retained_ || retiring() || (cancelled && cancelled())) {
      return false;
    }
    spec_ = std::move(spec);
    generation_ = ++next_generation;
    prepared_ = prepare ? prepare() : spec_.primary_binding && io_.prepare && io_.prepare(spec_);
    if (!prepared_ || (cancelled && cancelled()) || !io_.create) {
      return false;
    }
    // An Add can succeed even when publication times out. Record it before any validation.
    binding_ = io_.create(spec_);
    was_published_ = !binding_.display_name.empty();
    if ((cancelled && cancelled()) || !binding_.identity || binding_.display_name.empty() || binding_.device_path.empty()) {
      return false;
    }
    return !spec_.primary_binding || bind();
  }

  bool session_t::adopt(display_spec_t spec, creation_result_t binding, bool prepared) {
    if (owns_display() || prepared_ || retained_ || retiring()) {
      return false;
    }
    spec_ = std::move(spec);
    binding_ = std::move(binding);
    prepared_ = prepared;
    was_published_ = !binding_.display_name.empty();
    generation_ = ++next_generation;
    return true;
  }

  const display_spec_t &session_t::spec() const { return spec_; }
  const creation_result_t &session_t::binding() const { return binding_; }
  bool session_t::owns_display() const { return binding_.identity.has_value() && !retirement_complete_; }
  bool session_t::prepared() const { return prepared_; }
  bool session_t::was_published() const { return was_published_; }
  bool session_t::pause_requested() const { return pause_requested_; }
  bool session_t::paused() const { return paused_; }
  bool session_t::has_retained() const { return static_cast<bool>(retained_); }
  bool session_t::retiring() const { return retirement_.has_value(); }
  bool session_t::removal_accepted() const { return removal_accepted_; }
  bool session_t::retirement_complete() const { return retirement_complete_; }
  std::uint64_t session_t::generation() const { return generation_; }
  void session_t::set_exclusive(bool exclusive) { spec_.exclusive = exclusive; }
  void session_t::update_mode(std::uint32_t width, std::uint32_t height, std::uint32_t refresh_millihz) {
    spec_.width = width;
    spec_.height = height;
    spec_.refresh_millihz = refresh_millihz;
  }

  bool session_t::update_binding(creation_result_t binding) {
    if (!owns_display() || !binding.identity ||
        (!binding_.device_path.empty() && !binding.device_path.empty() && _wcsicmp(binding_.device_path.c_str(), binding.device_path.c_str()) != 0) ||
        (!same_identity(*binding_.identity, *binding.identity) &&
         (binding_.device_path.empty() || binding.device_path.empty()))) {
      return false;
    }
    if (binding.device_path.empty()) {
      binding.device_path = binding_.device_path;
    }
    if (!binding.render_adapter_luid) {
      binding.render_adapter_luid = binding_.render_adapter_luid;
    }
    was_published_ = was_published_ || !binding.display_name.empty();
    binding_ = std::move(binding);
    return true;
  }

  void session_t::invalidate_gdi() { binding_.display_name.clear(); }

  bool session_t::refresh() {
    if (!owns_display() || !io_.query) {
      invalidate_gdi();
      return false;
    }
    const auto observed = io_.query(binding_);
    if (observed.state != display_identity_state_e::present || observed.display_name.empty() || observed.device_path.empty()) {
      invalidate_gdi();
      return false;
    }
    auto current = binding_;
    current.device_path = observed.device_path;
    current.display_name = observed.display_name;
    current.friendly_name = observed.friendly_name;
    if (!update_binding(std::move(current))) {
      invalidate_gdi();
      return false;
    }
    return true;
  }

  bool session_t::bind() {
    return owns_display() && !binding_.device_path.empty() &&
           (!spec_.primary_binding || (io_.bind && io_.bind(binding_.device_path)));
  }

  bool session_t::promote() {
    return owns_display() && !retiring() && !binding_.device_path.empty() && io_.promote && io_.promote(binding_.device_path, spec_.exclusive);
  }

  bool session_t::restore() {
    if (retirement_complete_) {
      return true;
    }
    if (!prepared_ && !owns_display() && !retained_) {
      return true;
    }
    if (owns_display() && binding_.device_path.empty()) {
      if (!io_.query) {
        return false;
      }
      const auto observed = io_.query(binding_);
      if (observed.state == display_identity_state_e::indeterminate ||
          (observed.state == display_identity_state_e::present && observed.device_path.empty())) {
        return false;
      }
      if (!observed.device_path.empty()) {
        auto current = binding_;
        current.device_path = observed.device_path;
        current.display_name = observed.display_name;
        current.friendly_name = observed.friendly_name;
        if (!update_binding(std::move(current))) {
          return false;
        }
      }
    }
    if (!io_.restore || !io_.restore(binding_.device_path)) {
      return false;
    }
    prepared_ = false;
    return true;
  }

  bool session_t::pause(std::wstring_view local_sink) {
    if (retiring() || !owns_display() || binding_.device_path.empty()) {
      return false;
    }
    pause_requested_ = true;
    if (!paused_) {
      const auto sink = local_sink.empty() ? std::wstring_view(spec_.local_sink) : local_sink;
      paused_ = io_.pause && io_.pause(binding_.device_path, retained_, sink);
    }
    invalidate_gdi();
    return paused_;
  }

  bool session_t::begin_resume() {
    if (retiring() || !owns_display()) {
      return false;
    }
    if (!pause_requested_) {
      return true;
    }
    if (!paused_ && !pause()) {
      return false;
    }
    // An accepted mutation can still fail verification; the old pause proof is already invalid.
    paused_ = false;
    return retained_ && io_.reactivate && io_.reactivate(retained_, spec_.exclusive);
  }

  void session_t::commit_resume() {
    if (!retiring()) {
      // Mode, HDR and topology are final now. Earlier mutations can relocate the cursor again.
      if (retained_ && io_.restore_cursor) {
        io_.restore_cursor(retained_);
      }
      retained_.reset();
      pause_requested_ = paused_ = false;
    }
  }

  void session_t::begin_retirement(session_retirement_callbacks_t callbacks, session_retirement_policy_t policy) {
    if (retirement_ || retirement_complete_) {
      return;
    }
    retirement_callbacks_ = std::move(callbacks);
    retirement_policy_ = policy;
    retirement_.emplace(retirement_record_t {was_published_, io_.now ? io_.now() : time_point {}});
    next_remove_attempt_ = retirement_->started;
  }

  void session_t::invalidate_retirement_preparation() { retirement_prepared_ = false; }

  display_identity_state_e session_t::retirement_query() const {
    if (retirement_callbacks_.query) {
      return retirement_callbacks_.query(*this);
    }
    if (!binding_.identity) {
      return display_identity_state_e::absent;
    }
    if (retirement_policy_.require_device_absence) {
      return io_.query_retirement ? io_.query_retirement(binding_) : display_identity_state_e::indeterminate;
    }
    return io_.query ? io_.query(binding_).state : display_identity_state_e::indeterminate;
  }

  bool session_t::retire_until(time_point deadline) {
    if (retirement_complete_) {
      return true;
    }
    if (!io_.now || !io_.sleep) {
      return false;
    }
    if (!retirement_) {
      begin_retirement();
    }
    if (retirement_policy_.retry_acknowledged_removal_each_slice) {
      removal_accepted_ = false;
    }
    const auto prepare_once = [&]() {
      if (!retirement_prepared_) {
        retirement_prepared_ = retirement_callbacks_.prepare ? retirement_callbacks_.prepare(*this) : restore();
      }
      return retirement_prepared_;
    };
    while (!prepare_once()) {
      const auto now = io_.now();
      if (!retirement_policy_.retry_prepare_within_slice || now >= deadline || retirement_policy_.timing.poll_interval <= std::chrono::milliseconds::zero()) {
        return false;
      }
      io_.sleep(std::min(retirement_policy_.timing.poll_interval, std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)));
    }
    if (!binding_.identity) {
      // A failed Add never returned driver ownership. Its prepared baseline still needs cleanup,
      // but neither a driver query, a Remove nor late-publication quarantine applies.
      if (!cleanup_finished_) {
        if (retirement_callbacks_.finish && !retirement_callbacks_.finish(*this)) {
          return false;
        }
        cleanup_finished_ = true;
      }
      retirement_complete_ = true;
      prepared_ = pause_requested_ = paused_ = false;
      retained_.reset();
      return true;
    }
    struct clock_t {
      const session_io_t &io;
      time_point now() const { return io.now(); }
      void sleep_for(std::chrono::milliseconds delay) { io.sleep(delay); }
    } clock {io_};
    const retirement_callbacks_t callbacks {
      .before_observation = [&]() {
        if (!prepare_once()) {
          return retirement_step_e::blocked;
        }
        if (binding_.identity && !removal_accepted_ && clock.now() >= next_remove_attempt_) {
          if (retirement_callbacks_.before_remove && !retirement_callbacks_.before_remove(*this)) {
            retirement_prepared_ = false;
            return retirement_step_e::blocked;
          }
          removal_accepted_ = io_.remove && io_.remove(spec_.guid);
          next_remove_attempt_ = clock.now() + std::max(retirement_policy_.remove_retry_interval, std::chrono::milliseconds::zero());
        }
        return retirement_step_e::observe;
      },
      .query = [&]() { return retirement_query(); },
      .finish = [&]() {
        if (!cleanup_finished_) {
          if (retirement_callbacks_.finish && !retirement_callbacks_.finish(*this)) {
            return false;
          }
          cleanup_finished_ = true;
        }
        if (retirement_query() != display_identity_state_e::absent) {
          return false;
        }
        retirement_complete_ = true;
        prepared_ = pause_requested_ = paused_ = false;
        retained_.reset();
        return true;
      },
    };
    // finish changes this owner but never destroys it. Keep the timing proof independent.
    const auto proof = *retirement_;
    return proof.wait_until(deadline, retirement_policy_.timing, callbacks, clock);
  }
}  // namespace VDISPLAY

/**
 * @file src/platform/windows/text_game_pin_controller.cpp
 * @brief What the Host AI path observes of a Game 3D text game and its helper, and the pin of
 *        the helper's text in the Host warp.
 */
#include "text_game_pin_controller.h"
#include "game3d_win32_util.h"

// standard includes
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <utility>

// platform includes
#include <windows.h>

// local includes
#include "src/logging.h"

namespace platf::dxgi {
  using namespace std::literals;

  namespace {
    /** When the process with this id was created, or zero when it cannot be opened. */
    std::uint64_t process_creation_time(const std::uint32_t process_id) {
      return platf::game3d::win32::process_id_creation_time(process_id);
    }
  }  // namespace

  std::optional<text_game_pin_controller_t::identity_t> text_game_pin_controller_t::observe(
    const time_point_t now,
    const foreground_window::rect_t &screen
  ) {
    return observe(now, screen, foreground_window::sample, process_creation_time);
  }

  std::optional<text_game_pin_controller_t::identity_t> text_game_pin_controller_t::observe(
    const time_point_t now,
    const foreground_window::rect_t &screen,
    const std::function<foreground_window::observation_t()> &sample_foreground,
    const std::function<std::uint64_t(std::uint32_t)> &creation_time_of
  ) {
    if (checked_at_ != time_point_t {} && now - checked_at_ < 250ms) {
      return source_;
    }
    checked_at_ = now;
    const auto previous = std::exchange(source_, std::nullopt);
    const auto foreground = sample_foreground();
    const auto &client = foreground.client_screen_rect;
    // While its window is dragged or resized, and for a sample that could not be completed,
    // the game is still the one in front: only where its picture lies is unknown. It stays
    // converted and its text is left unpinned, instead of the session dropping to flat eyes
    // and reporting the game gone for the length of a drag.
    const bool geometry_pending = foreground.status == foreground_window::status_e::interactive_move_size ||
                                  foreground.status == foreground_window::status_e::geometry_unavailable ||
                                  foreground.status == foreground_window::status_e::foreground_changed;
    if (geometry_pending && previous && foreground.process_id == previous->process_id) {
      client_ = {};
      source_ = previous;
      return source_;
    }
    std::uint64_t creation_time = 0u;
    if (foreground_window::carries_geometry(foreground.status) && foreground.process_id && client.valid() &&
        client.left >= screen.left && client.top >= screen.top &&
        client.right <= screen.right && client.bottom <= screen.bottom) {
      creation_time = creation_time_of(foreground.process_id);
    }
    // The owner's approval for this very process decides. Its helper adds the game's own text
    // where it runs; a game without one is converted all the same.
    if (creation_time == 0u || !platf::game3d::text_hook::consented(foreground.process_id, creation_time)) {
      // The reader's own handles would keep an exited game's sections alive.
      reader_.close();
      coverage_uploaded_ = false;
      return std::nullopt;
    }
    client_ = {client.left - screen.left, client.top - screen.top, client.right - screen.left, client.bottom - screen.top};
    window_ = foreground.window;
    const identity_t identity {foreground.process_id, creation_time, 0u};
    if (!helper_identity_ || *helper_identity_ != identity) {
      // Nothing learned about one game, or still on its way back from the GPU, is another's.
      helper_identity_ = identity;
      helper_sees_text_ = false;
      text_shown_ = false;
      unshown_since_.reset();
      fallback_ = {};
      protection_ = {};
      verdict_staged_ = false;
      coverage_uploaded_ = false;
      band_ = {};
    }
    source_ = identity;
    return source_;
  }

  void text_game_pin_controller_t::restart_session() {
    source_.reset();
    checked_at_ = {};
    route_ = route_e::none;
  }

  bool text_game_pin_controller_t::switch_route(const bool text_game) {
    const auto route = text_game ? route_e::depth : route_e::external_or_flat;
    if (route == route_) {
      return false;
    }
    route_ = route;
    return true;
  }

  bool text_game_pin_controller_t::take_text(
    const bool answered,
    const platf::game3d::text_hook::text_t &text,
    const time_point_t now
  ) {
    namespace hook = platf::game3d::text_hook;
    // Any refusal from here on that names no reason of its own is the host's.
    fallback_ = hook::fallback_e::unavailable;
    // A helper that knows its window reports for that one only: the game's own dialog in front
    // is not the picture its text was drawn on. Window handles are 32 bits in either process.
    if (!answered || text.unsure || text.top >= text.bottom || (text.window != 0u && static_cast<std::uint32_t>(text.window) != static_cast<std::uint32_t>(window_))) {
      fallback_ = !answered ? hook::fallback_e::no_helper : text.unsure ? hook::fallback_e::unsure :
                                                                         hook::fallback_e::no_text;
      text_shown_ = false;
      // A verdict still on its way is of a draw made before this: it is not taken later.
      verdict_staged_ = false;
      if (answered && text.unsure) {
        // The helper itself asks for the picture-based search: no waiting for its text to age.
        helper_sees_text_ = false;
      } else {
        not_shown(now);
      }
      return false;
    }
    return true;
  }

  bool text_game_pin_controller_t::prepare(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    const D3D11_TEXTURE2D_DESC &colour,
    const float sdr_white,
    std::array<UINT, 4u> &shape
  ) {
    namespace hook = platf::game3d::text_hook;
    const auto &client = client_;
    if (!source_ || !client.valid() || client.left < 0 || client.top < 0 ||
        client.right > static_cast<LONG>(colour.Width) || client.bottom > static_cast<LONG>(colour.Height)) {
      return false;
    }
    hook::text_t text;
    const bool answered = reader_.read(source_->process_id, source_->creation_time, text);
    if (!take_text(answered, text, std::chrono::steady_clock::now())) {
      return false;
    }
    D3D11_TEXTURE2D_DESC current {};
    if (coverage_texture_) {
      coverage_texture_->GetDesc(&current);
    }
    if (!coverage_srv_ || current.Width != text.surface_width || current.Height != text.surface_height) {
      coverage_srv_.reset();
      coverage_texture_.reset();
      coverage_uploaded_ = false;
      D3D11_TEXTURE2D_DESC desc {};
      desc.Width = text.surface_width;
      desc.Height = text.surface_height;
      desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1u;
      desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      if (FAILED(device->CreateTexture2D(&desc, nullptr, &coverage_texture_)) ||
          FAILED(device->CreateShaderResourceView(coverage_texture_.get(), nullptr, &coverage_srv_))) {
        coverage_srv_.reset();
        coverage_texture_.reset();
        return false;
      }
    }
    const auto now = std::chrono::steady_clock::now();
    if (!coverage_uploaded_ || coverage_sequence_ != text.sequence) {
      // Only the rows with text are read from the game and only those rows are consulted below.
      const D3D11_BOX rows {0u, text.top, 0u, text.surface_width, text.bottom, 1u};
      const UINT pitch = text.surface_width * 4u;
      context->UpdateSubresource(coverage_texture_.get(), 0u, &rows, text.pixels + static_cast<std::size_t>(text.top) * pitch, pitch, 0u);
      // The game may have drawn meanwhile. Those pixels are then a mix of two moments: usable
      // for this frame, and read again for the next one.
      coverage_uploaded_ = true;
      coverage_sequence_ = reader_.still_current(text) ? text.sequence : text.sequence - 1u;
    }
    const float scale_y = static_cast<float>(client.bottom - client.top) / static_cast<float>(text.surface_height);
    models::host_sbs_shared_ui_warp::text_game_constants_t constants;
    // The coverage picture was read for the surface rows [top, bottom) only, and a source row
    // samples the two surface rows around its centre. The source rows kept are those whose
    // sample lies wholly inside the rows that were read, at any scale of the picture.
    const float text_first = std::ceil(static_cast<float>(client.top) + (static_cast<float>(text.top) + 0.5f) * scale_y - 0.5f);
    const float text_end = std::ceil(static_cast<float>(client.top) + (static_cast<float>(text.bottom) - 0.5f) * scale_y - 0.5f);
    if (!(text_end > text_first) || text_first < 0.0f) {
      return false;
    }
    constants.text_top = static_cast<std::uint32_t>(text_first);
    constants.text_bottom = static_cast<std::uint32_t>(text_end);
    // The scene's positive display bound in the scale of the game's picture.
    constants.front_cap = 0.04f * static_cast<float>(client.right - client.left) / static_cast<float>(colour.Width);
    // What SDR white is in an HDR capture, so that the picture compares with the game's colours.
    constants.sdr_white = sdr_white;
    // The pinned rows are the text and as far above and below it as the distance rule can still
    // bound the scene. The field lies within the display bound of zero and the plane within
    // the policy's ceiling, so they differ by at most their sum, which the rule's slope of half
    // a pixel per pixel covers in twice that many pixels.
    const float reach = 2.0f * (constants.front_cap + sunshine_game3d::ui_adaptive::comfort_cap_uv) * static_cast<float>(colour.Width) + 2.0f;
    const auto first = static_cast<LONG>(std::floor(text_first - reach)) - 1;
    const auto last = static_cast<LONG>(std::ceil(text_end + reach)) + 1;
    UINT band_top = static_cast<UINT>(std::clamp<LONG>(first, client.top, client.bottom)) / 16u * 16u;
    UINT band_end = std::min<UINT>(colour.Height, (static_cast<UINT>(std::clamp<LONG>(last, client.top, client.bottom)) + 15u) / 16u * 16u);
    if (band_top >= band_end) {
      return false;
    }
    // The band grows while the game's picture stays where it is: one, two and three lines of
    // dialogue then share one set of pin planes instead of reallocating them line by line. A
    // band more than twice what the text has needed for some seconds, as a page of text or a
    // menu leaves behind, is given up, so that one such page does not cost every later frame.
    if (band_.client == client && band_.width == colour.Width && band_.height == colour.Height) {
      const bool oversized = (band_end - band_top) * 2u < band_.end - band_.top;
      if (!oversized) {
        band_needed_at_ = now;
      }
      if (!oversized || now - band_needed_at_ < std::chrono::seconds(3)) {
        band_top = std::min(band_top, band_.top);
        band_end = std::max(band_end, band_.end);
      }
    } else {
      band_needed_at_ = now;
    }
    band_ = {client, colour.Width, colour.Height, band_top, band_end};
    shape = {colour.Width, colour.Height, band_top, band_end - band_top};
    constants.left = static_cast<float>(client.left);
    constants.top = static_cast<float>(client.top);
    constants.inverse_width = 1.0f / static_cast<float>(client.right - client.left);
    constants.inverse_height = 1.0f / static_cast<float>(client.bottom - client.top);
    constants.enabled = 1u;
    // The policy holds the text at screen depth and moves it forward only as far as the scene
    // under it requires.
    auto &live = plane_live_;
    live = {};
    live.now_ms = live.tick_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
    live.epoch = 1u;
    live.source_id = (static_cast<std::uint64_t>(source_->process_id) << 32u) ^ source_->creation_time;
    live.sequence = ++plane_sequence_;
    live.viewport = (static_cast<std::uint32_t>(client.right - client.left) << 16u) ^ static_cast<std::uint32_t>(client.bottom - client.top);
    live.eligible = true;
    live.front_cap_uv = constants.front_cap;
    constants.plane = plane_.prepare(live).applied_uv;
    if (!pin_cbuffer_) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = sizeof(constants);
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      const D3D11_SUBRESOURCE_DATA initial {&constants, 0u, 0u};
      return SUCCEEDED(device->CreateBuffer(&desc, &initial, &pin_cbuffer_));
    }
    context->UpdateSubresource(pin_cbuffer_.get(), 0u, nullptr, &constants, 0u, 0u);
    return true;
  }

  text_game_pin_controller_t::pin_draw_t text_game_pin_controller_t::queue(
    ID3D11Device *device,
    ID3D11DeviceContext *context,
    const models::host_sbs_shared_ui_warp::program_t &program,
    models::host_sbs_shared_ui_warp::request_t request,
    const std::array<UINT, 4u> &shape,
    const bool selected,
    const bool diagnostics
  ) {
    pin_draw_t result;
    request.text_game = true;
    request.text_game_coverage = coverage_srv_.get();
    request.text_game_constants = pin_cbuffer_.get();
    // Text the picture does not show is only observed: one pass over its rows.
    const bool observe_only = !selected;
    request.text_game_observe_only = observe_only;
    std::string error;
    const bool queued = shared_warp_.ensure(device, shape, error) &&
                        shared_warp_.queue(context, program, request, error);
    result.selected = selected && queued && shared_warp_.matches(context, request);
    if (result.selected) {
      result.views = shared_warp_.views();
    }
    if (queued) {
      service_verdict(device, context);
    }
    if (diagnostics) {
      // A draw on the pinned route whose text the picture did not show at the latest finding
      // counts as observed: the GPU leaves such a frame unpinned.
      ++(result.selected && text_shown_ ? pin_draws_ : queued ? pin_probes_ : pin_misses_);
      const auto log_now = std::chrono::steady_clock::now();
      if (log_now - logged_at_ >= std::chrono::seconds(5)) {
        logged_at_ = log_now;
        BOOST_LOG(info) << "Game 3D text pin: draws="sv << pin_draws_ << " observed="sv << pin_probes_ << " failed="sv << pin_misses_
                        << " rows="sv << shape[2] << '+' << shape[3]
                        << " plane="sv << plane_.current().applied_uv << '>' << plane_.current().target_uv
                        << '(' << plane_.current().status << ')'
                        << " shown light="sv << verdict_[1] << '/' << verdict_[0]
                        << " dark="sv << verdict_[3] << '/' << verdict_[2]
                        << (error.empty() ? ""s : " error="s + error);
        pin_draws_ = pin_probes_ = pin_misses_ = 0u;
      }
    }
    return result;
  }

  void text_game_pin_controller_t::take_verdict(const std::array<std::uint32_t, 4u> &verdict, const time_point_t now) {
    verdict_ = verdict;
    // The same rule as the shader's. The text is pinned only while the picture shows it in
    // the helper's colours, and only text it has shown once proves that the helper serves
    // this game; until then the picture-based search and its pin stay.
    const auto kind_shown = [](const std::uint32_t cores, const std::uint32_t shown) {
      return cores < 16u || shown * 16u >= cores;
    };
    text_shown_ = verdict_[0] + verdict_[2] >= 16u && kind_shown(verdict_[0], verdict_[1]) &&
                  kind_shown(verdict_[2], verdict_[3]);
    if (text_shown_) {
      helper_sees_text_ = true;
      unshown_since_.reset();
    } else {
      not_shown(now);
    }
  }

  void text_game_pin_controller_t::service_verdict(ID3D11Device *device, ID3D11DeviceContext *context) {
    namespace adaptive = sunshine_game3d::ui_adaptive;
    D3D11_MAPPED_SUBRESOURCE mapped {};
    if (verdict_staged_ && SUCCEEDED(context->Map(verdict_staging_.get(), 0u, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped))) {
      const auto *words = static_cast<const std::uint32_t *>(mapped.pData);
      std::array<std::uint32_t, 4u> verdict;
      std::memcpy(verdict.data(), words, sizeof(verdict));
      take_verdict(verdict, std::chrono::steady_clock::now());
      // The policy asks for at most one observation every 100 ms.
      const auto &origin = plane_staged_;
      if (origin.eligible && origin.tick_ms >= plane_observed_ms_ + adaptive::probe_interval_ms) {
        adaptive::sample sample;
        sample.provenance = origin;
        sample.width = origin.viewport >> 16u;
        sample.height = origin.viewport & 0xffffu;
        for (unsigned index = 0u; index < sample.tiles.size(); ++index) {
          const unsigned x = index % 16u, y = index / 16u;
          auto &tile = sample.tiles[index];
          tile.pixels = ((x + 1u) * sample.width / 16u - x * sample.width / 16u) * ((y + 1u) * sample.height / 16u - y * sample.height / 16u);
          const auto *counts = words + 4u + index * 6u;
          // Tiles are counted in source pixels and sized here by integer division, so a count
          // can exceed its tile by the rounding at a tile's edge. Scale all six alike; only
          // their ratios are judged.
          const std::uint32_t covered = counts[0];
          const double scale = covered > tile.pixels && covered != 0u ? static_cast<double>(tile.pixels) / covered : 1.0;
          tile.covered = static_cast<std::uint32_t>(covered * scale);
          std::uint32_t previous = tile.covered;
          for (unsigned level = 0u; level < tile.bad.size(); ++level) {
            previous = tile.bad[level] = std::min(previous, static_cast<std::uint32_t>(counts[1u + level] * scale));
          }
        }
        const auto read_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
        if (plane_.observe(sample, read_ms)) {
          plane_observed_ms_ = origin.tick_ms;
        }
      }
      context->Unmap(verdict_staging_.get(), 0u);
      verdict_staged_ = false;
    }
    if (verdict_staged_) {
      return;
    }
    if (!verdict_staging_) {
      D3D11_BUFFER_DESC desc {};
      desc.ByteWidth = static_cast<UINT>(models::host_sbs_shared_ui_warp::text_game_verdict_words * 4u);
      desc.Usage = D3D11_USAGE_STAGING;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      if (FAILED(device->CreateBuffer(&desc, nullptr, &verdict_staging_))) {
        return;
      }
    }
    if (auto *verdict = shared_warp_.text_game_verdict()) {
      context->CopyResource(verdict_staging_.get(), verdict);
      verdict_staged_ = true;
      plane_staged_ = plane_live_;
    }
  }

  void text_game_pin_controller_t::not_shown(const time_point_t now) {
    if (!unshown_since_) {
      unshown_since_ = now;
    }
  }

  bool text_game_pin_controller_t::helper_serves(const time_point_t now) const {
    return helper_sees_text_ && (!unshown_since_ || now - *unshown_since_ <= std::chrono::seconds(3));
  }

  bool text_game_pin_controller_t::picture_search_covers(const std::uint32_t process_id, const time_point_t now) const {
    return !helper_serves(now) && source_ && source_->process_id == process_id;
  }

  void text_game_pin_controller_t::note_protection(const bool usable, const bool prepared, const bool pinned, const time_point_t now) {
    namespace hook = platf::game3d::text_hook;
    if (!source_) {
      return;
    }
    const bool helper = pinned || helper_serves(now);
    const hook::protection_t state {
      helper ? hook::protection_e::game_text : hook::protection_e::picture,
      helper ? hook::fallback_e::none : !usable ? hook::fallback_e::unavailable :
                                      prepared  ? hook::fallback_e::not_shown :
                                                  fallback_,
    };
    // Told at once when it changes, which a change of the game in front makes it do.
    if (state != protection_ || now - protection_at_ >= std::chrono::seconds(1)) {
      protection_ = state;
      protection_at_ = now;
      hook::report_protection(source_->process_id, source_->creation_time, state);
    }
  }

  void text_game_pin_controller_t::reset() {
    reader_.close();
    helper_sees_text_ = false;
    text_shown_ = false;
    helper_identity_.reset();
    band_ = {};
    coverage_srv_.reset();
    coverage_texture_.reset();
    coverage_uploaded_ = false;
    pin_cbuffer_.reset();
    verdict_staging_.reset();
    verdict_staged_ = false;
    plane_.reset();
    plane_observed_ms_ = 0u;
    shared_warp_ = {};
  }
}  // namespace platf::dxgi

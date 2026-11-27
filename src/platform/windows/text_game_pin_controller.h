/**
 * @file src/platform/windows/text_game_pin_controller.h
 * @brief What the Host AI path observes of a Game 3D text game and its helper, and the pin of
 *        the helper's text in the Host warp.
 */
#pragma once

// standard includes
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>

// platform includes
#include <d3d11.h>

// local includes
#include "display.h"
#include "foreground_window_region.h"
#include "game3d_text_hook.h"
#include "src/game_source_tracker.h"
#include "src/host_sbs_shared_ui_warp_runtime.h"
#include "tools/reshade/game3d_ui_adaptive.h"

namespace platf::dxgi {
  /**
   * The text-game pin of one converter.
   *
   * A Game 3D text game has no depth and publishes no stereo of its own. It supplies where its
   * text is, through the text helper loaded into it, and the host supplies the rest: while such
   * a game is in front, packed Game 3D converts the captured picture with the Host AI path.
   *
   * This owns what that path observes of the game and its helper: the approved game in front,
   * the packed route, the helper's text and whether the picture shows it, the pinned rows, the
   * text plane, and the device objects of the pass that observes and pins the text in the Host
   * warp. The converter decides whether a frame takes the Host AI path, records the draw, and
   * owns everything a route switch resets.
   */
  class text_game_pin_controller_t {
  public:
    using identity_t = ::video::game_source_identity_t;
    using time_point_t = std::chrono::steady_clock::time_point;

    /**
     * The text game in front on this display, as a game source identity: an application its
     * owner approved for conversion. The foreground is sampled a few times a second; the stream
     * converts at up to its frame rate. The game is in front also while a dialog of its own lies
     * on its picture, and while its picture reaches slightly past this display
     * (foreground_window::sample_picture_root, foreground_window::visible_on_output).
     *
     * @param screen Where this display is on the Windows desktop.
     */
    std::optional<identity_t> observe(time_point_t now, const foreground_window::rect_t &screen);

    /** observe(), with the foreground sample and a process's creation time supplied. */
    std::optional<identity_t> observe(
      time_point_t now,
      const foreground_window::rect_t &screen,
      const std::function<foreground_window::observation_t()> &sample_foreground,
      const std::function<std::uint64_t(std::uint32_t)> &creation_time
    );

    /** A new session: no game is known to be in front, and no packed route is taken yet. */
    void restart_session();

    /**
     * Take the packed route of this frame: the Host AI path for a text game, or the export or flat
     * path. True when it differs from the route of the previous frame, and the converter then
     * resets what one route must never reuse from the other.
     */
    bool switch_route(bool text_game);

    /**
     * Bring the text the game's helper reports onto this device and describe where it lies in the
     * source about to be drawn. Returns false, changing nothing that is drawn, whenever there is
     * no text or any part of it cannot be trusted; the draw then keeps its ordinary geometry.
     *
     * @param colour The source about to be drawn.
     * @param sdr_white What SDR white is in an HDR capture, in scRGB units.
     * @param shape Set to the full source size and the band of pinned rows.
     */
    bool prepare(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      const D3D11_TEXTURE2D_DESC &colour,
      float sdr_white,
      std::array<UINT, 4u> &shape
    );

    /**
     * The latest observation found the helper's text in the picture: it is pinned. Otherwise the
     * helper's report is only observed, and the draw keeps its ordinary geometry.
     */
    [[nodiscard]] bool text_shown() const noexcept {
      return text_shown_;
    }

    /**
     * Whether a draw takes the pinned route. That route decides on the GPU, from its own
     * picture, whether this very frame is pinned; text_shown() is the answer of a draw a frame
     * or two older. While the helper serves the game the route is therefore kept through a
     * finding that the text is not shown: chosen from the older answer alone, each frame that
     * hid the text would unpin the frames that follow it as well, and the text would come back
     * as late again.
     */
    [[nodiscard]] bool pin_route(const time_point_t now) const {
      return text_shown_ || helper_serves(now);
    }

    /** The pass of one prepared draw, and the pin planes when the draw uses them. */
    struct pin_draw_t {
      bool selected = false;  ///< The draw uses the pin.
      models::host_sbs_shared_ui_warp::views_t views;
    };

    /**
     * Queue the pass that observes the prepared text and, for a draw that pins it, writes the pin
     * planes. Then collect the verdict of an earlier draw if it is ready, and stage this one's.
     * The converter records its draw next: preparation bindings are left clear for it.
     *
     * @param request The draw's operands; the text-game fields are filled here.
     * @param shape As prepare() set it.
     * @param selected Whether the draw is to use the pin.
     * @param diagnostics Count the passes and log them every five seconds.
     */
    pin_draw_t queue(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      const models::host_sbs_shared_ui_warp::program_t &program,
      models::host_sbs_shared_ui_warp::request_t request,
      const std::array<UINT, 4u> &shape,
      bool selected,
      bool diagnostics
    );

    /**
     * Take the GPU's verdict on an earlier draw: light glyph cores and of them those the picture
     * shows, then dark ones and those shown.
     */
    void take_verdict(const std::array<std::uint32_t, 4u> &verdict, time_point_t now);

    /**
     * The helper's text counts as serving the game from the first time the picture shows it, and
     * until three seconds of findings that it does not: a line that fades or a pause between two
     * lines keeps it. Past that the picture-based search covers the game again, as it does before
     * the helper has shown anything, so a helper that loses the text, or is not sure of it,
     * costs no protection. Only findings count: a still picture, which is not looked at, says
     * nothing against the helper.
     */
    [[nodiscard]] bool helper_serves(time_point_t now) const;

    /**
     * Whether the picture-based subtitle search covers this foreground process as the text game
     * in front: it is that game, and its helper has not lately shown that it sees the text.
     */
    [[nodiscard]] bool picture_search_covers(std::uint32_t process_id, time_point_t now) const;

    /** Tell the setup card what protects the text of the game in front. */
    void note_protection(bool usable, bool prepared, bool pinned, time_point_t now);

    /** Release the device objects and forget everything learned about the helper. */
    void reset();

  private:
    enum class route_e : std::uint8_t {
      none,
      depth,  ///< Host AI path for a text game.
      external_or_flat,  ///< An add-on export, or duplicate flat eyes without one.
    };

    struct band_t {
      foreground_window::rect_t client {};
      UINT width = 0u, height = 0u;
      UINT top = 0u, end = 0u;
    };

    /** The helper's answer for a draw. False, with the reason kept, when there is no usable text. */
    bool take_text(bool answered, const platf::game3d::text_hook::text_t &text, time_point_t now);

    /** Collect the verdict of an earlier draw if it is ready, and stage the one just queued. */
    void service_verdict(ID3D11Device *device, ID3D11DeviceContext *context);

    /** One more finding that the picture does not show the helper's text, or that it has none. */
    void not_shown(time_point_t now);

    std::optional<identity_t> source_;
    time_point_t checked_at_ {};
    route_e route_ = route_e::none;
    // Where the text game's picture is in the captured source, and the text its helper reports.
    // The rectangle is empty while the game is in front but its place is not known. It is the
    // whole picture: it may reach slightly past the source and is never cut to it.
    foreground_window::rect_t client_ {};
    std::uintptr_t window_ = 0u;  ///< The game's window in front.
    band_t band_;  ///< The pinned rows used so far for this picture.
    time_point_t band_needed_at_ {};  ///< When the text last needed half of them.
    // The picture has shown text in the colours the helper reported, so the helper serves this
    // game. From then on the picture-based subtitle search has nothing to add and is not run for
    // it: it costs more GPU time than the depth model itself.
    bool helper_sees_text_ = false;
    bool text_shown_ = false;  ///< See text_shown().
    /// Since when every finding has been that the picture does not show the helper's text.
    std::optional<time_point_t> unshown_since_;
    platf::game3d::text_hook::fallback_e fallback_ {};  ///< Why the helper's text was not prepared, last time.
    platf::game3d::text_hook::protection_t protection_ {};  ///< As last told to the setup card.
    time_point_t protection_at_ {};
    std::optional<identity_t> helper_identity_;
    platf::game3d::text_hook::reader_t reader_;
    texture2d_t coverage_texture_;
    shader_res_t coverage_srv_;
    std::uint32_t coverage_sequence_ = 0u;
    bool coverage_uploaded_ = false;
    buf_t pin_cbuffer_;
    // The GPU's observation of each draw with text, read back a frame or two late and never
    // waited for: whether the picture shows the text, which selects the pin for the following
    // draws (the shader of a pinned draw decides it again for that very frame), and the scene
    // under the text, which places the text plane.
    buf_t verdict_staging_;
    bool verdict_staged_ = false;
    std::array<std::uint32_t, 4u> verdict_ {};  ///< light cores, of them shown, dark cores, of them shown
    // Where the text sits in depth: the Game 3D add-on's own UI plane policy, fed with the scene
    // under the glyphs. Screen depth unless the scene there is nearer; then one of its levels.
    sunshine_game3d::ui_adaptive::policy plane_;
    sunshine_game3d::ui_adaptive::source plane_live_;  ///< Provenance of the draw being prepared.
    sunshine_game3d::ui_adaptive::source plane_staged_;  ///< Provenance of the staged observation.
    std::uint64_t plane_sequence_ = 0u;
    std::uint64_t plane_observed_ms_ = 0u;
    time_point_t logged_at_ {};
    unsigned pin_draws_ = 0u;
    unsigned pin_probes_ = 0u;
    unsigned pin_misses_ = 0u;
    models::host_sbs_shared_ui_warp::resources_t shared_warp_;
  };
}  // namespace platf::dxgi

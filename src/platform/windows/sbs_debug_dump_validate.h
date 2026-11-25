/**
 * @file src/platform/windows/sbs_debug_dump_validate.h
 * @brief Dump 3D record validators: pure checks of captured bytes, without D3D work or file I/O.
 */
#pragma once

// platform includes
#include <dxgiformat.h>

// standard includes
#include <cstddef>
#include <cstdint>
#include <vector>

namespace platf::sbs_debug {

  struct frame;

  namespace detail {
    /** Word `index` of a captured little-endian record. */
    std::uint32_t subtitle_word(
      const std::vector<std::uint8_t> &bytes,
      std::size_t index
    );

    /** The two words from `low_index` of a captured little-endian record, low word first. */
    std::uint64_t subtitle_u64(
      const std::vector<std::uint8_t> &bytes,
      std::size_t low_index
    );

    /** Pure exact-frame OCR8/UIP4 validation used before diagnostic publication. */
    bool subtitle_records_match_frame(
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &locator,
      const frame &completed
    );

    /** Accept exact-current evidence (including cadence-due reuse), or an authenticated held tuple. */
    bool subtitle_records_match_completion(
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &locator,
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    );

    bool subtitle_ocr_record_is_canonical_for_frame(
      const std::vector<std::uint8_t> &ocr,
      const frame &completed
    );

    /** Validate one optional diagnostic completion ring and require the dump's matched root. */
    bool gpu_trace_ring_is_canonical(
      const std::vector<std::uint8_t> &ring,
      const frame &completed
    );

    /** Authenticate the state/statistics pair before serializing V2 diagnostic documents. */
    bool parallax_v2_state_matches_frame(
      const std::vector<float> &state,
      const std::vector<float> &frame_stats,
      const frame &completed
    ) noexcept;

    /** Bind a high-grid dump to the exact schema-2 fused composite runtime identity. */
    bool composite_runtime_provenance_is_authenticated(const frame &completed) noexcept;

    enum class subtitle_contour_texture_e : std::uint8_t {
      support, glyph_matte, field_pin, distance, history, source, gaussian_weight, gaussian_blend, gaussian_vertical_conditioned,
    };

    /** Exact texture ABI, including the canonical 1x1 unsupported-source variant. */
    bool subtitle_contour_texture_layout_is_authenticated(
      subtitle_contour_texture_e role,
      DXGI_FORMAT format,
      std::uint32_t width,
      std::uint32_t height,
      std::size_t row_bytes,
      std::size_t byte_count,
      const frame &completed
    ) noexcept;

    /** Join SCT5 to already-authenticated OCR8/UIP4 and the completed geometry owner. */
    bool subtitle_contour_identity_matches_completion(
      const std::vector<std::uint8_t> &identity,
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &locator,
      const std::vector<float> &state,
      const frame &completed
    ) noexcept;

    /** Fail closed when contour authority or the independent detector snapshot gate abstains. */
    bool subtitle_contour_clear_policy_is_authenticated(
      const std::vector<std::uint8_t> &identity,
      const std::vector<std::uint8_t> &ocr,
      const std::vector<float> &state,
      const std::vector<std::uint8_t> &support,
      const std::vector<std::uint8_t> &field_pin,
      const std::vector<std::uint8_t> &distance,
      const std::vector<std::uint8_t> &history,
      const std::vector<std::uint8_t> &source,
      const std::vector<std::uint8_t> &probability
    ) noexcept;

    /** Authenticate the literal Gaussian proposal and the actual shared-conditioner leaves. */
    bool subtitle_gaussian_fields_are_authenticated(
      const std::vector<std::uint8_t> &identity,
      const std::vector<std::uint8_t> &locator,
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &distance,
      const std::vector<std::uint8_t> &weight,
      const std::vector<std::uint8_t> &blend,
      const std::vector<std::uint8_t> &vertical,
      const std::vector<std::uint8_t> &control,
      const std::vector<std::uint8_t> &base,
      const std::vector<std::uint8_t> &final
    ) noexcept;

    /** Replay the same-observation native matte and bind separate SCP4 without relying on a trace. */
    bool subtitle_analytic_tuple_is_authenticated(
      const std::vector<std::uint8_t> &identity,
      const std::vector<std::uint8_t> &locator,
      const std::vector<std::uint8_t> &ocr,
      const std::vector<std::uint8_t> &support,
      const std::vector<std::uint8_t> &source,
      const std::vector<std::uint8_t> &matte,
      const std::vector<std::uint8_t> &condition
    ) noexcept;

    /** Authenticate the optional OCR8/UIP4 and complete contour resource set. */
    bool subtitle_capture_resources_are_authenticated(const frame &completed) noexcept;
  }  // namespace detail
}  // namespace platf::sbs_debug

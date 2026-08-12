#pragma once

#include "config.h"
#include "depth_coordinate_v2.h"
#include "model_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

namespace models {

  // The exact-area RGB preprocess performs work proportional to source pixels so thin source
  // features survive the large resize into the depth grid. Bound that work to the largest shipped
  // Moonlight 3D raster (5K ultrawide, in either orientation); arbitrary 8K/16K inputs must not
  // silently turn one Host SBS frame into tens of millions of transfer-function evaluations.
  inline constexpr std::uint32_t host_sbs_v2_max_source_long_side = 5120u;
  inline constexpr std::uint64_t host_sbs_v2_max_source_pixels = 5120ull * 2160ull;

  struct depth_tensor_shape_t {
    int width = 0;
    int height = 0;

    constexpr bool valid() const noexcept {
      return width > 0 && height > 0;
    }

    constexpr bool operator==(const depth_tensor_shape_t &) const = default;
  };

  /** Half-open source-pixel rectangle selected for video-only depth inference. */
  struct depth_source_rect_t {
    std::uint32_t left = 0u;
    std::uint32_t top = 0u;
    std::uint32_t right = 0u;
    std::uint32_t bottom = 0u;

    constexpr std::uint32_t width() const noexcept {
      return right > left ? right - left : 0u;
    }

    constexpr std::uint32_t height() const noexcept {
      return bottom > top ? bottom - top : 0u;
    }

    constexpr bool valid() const noexcept {
      return width() > 0u && height() > 0u;
    }

    constexpr bool operator==(const depth_source_rect_t &) const = default;
  };

  struct depth_video_region_plan_t {
    depth_source_rect_t source_rect {};
    depth_tensor_shape_t tensor_shape {};
    float trimmed_area_fraction = 0.0f;

    constexpr bool operator==(const depth_video_region_plan_t &) const = default;
  };

  // A small inward trim may exclude a Chromium/player frame without materially cropping the
  // picture. Larger adaptation is rejected so the ordinary full-frame route remains the fallback.
  inline constexpr float host_sbs_v2_max_video_region_trim_fraction = 0.02f;


  namespace host_sbs_resolution_detail {
    inline int round_to_patch(const float value, const int patch = 14) {
      return std::max(patch, static_cast<int>(std::round(value / patch)) * patch);
    }

    inline std::pair<int, int> aspect_aligned_dims(
      float aspect,
      const int short_side,
      int max_width,
      int max_height,
      const int patch = 14
    ) {
      aspect = std::max(aspect, 1.0e-6f);
      max_width = std::max(patch, (max_width / patch) * patch);
      max_height = std::max(patch, (max_height / patch) * patch);
      const int requested_short = round_to_patch(static_cast<float>(short_side), patch);
      if (aspect >= 1.0f) {
        for (int height = std::min(requested_short, max_height);
             height >= patch;
             height -= patch) {
          const int width = round_to_patch(static_cast<float>(height) * aspect, patch);
          if (width <= max_width) {
            return {width, height};
          }
        }
      } else {
        for (int width = std::min(requested_short, max_width);
             width >= patch;
             width -= patch) {
          const int height = round_to_patch(static_cast<float>(width) / aspect, patch);
          if (height <= max_height) {
            return {width, height};
          }
        }
      }
      return {patch, patch};
    }
  }  // namespace host_sbs_resolution_detail

  /** Fit one patch-aligned depth tensor to a source aspect within native/profile bounds. */
  inline depth_tensor_shape_t fit_depth_tensor_shape(
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const int short_side,
    const float max_aspect
  ) {
    if (source_width == 0u || source_height == 0u) {
      return {};
    }

    const float aspect = static_cast<float>(source_width) /
                         static_cast<float>(source_height);
    const float bounded_max_aspect = std::max(1.0f, max_aspect);
    const float fitted_aspect = aspect >= 1.0f ?
                                  std::min(aspect, bounded_max_aspect) :
                                  1.0f / std::min(1.0f / aspect, bounded_max_aspect);
    const int bounded_short_side = std::clamp(short_side, 14, depth_engine_max_dim);
    const int max_width = static_cast<int>(std::min<std::uint32_t>(
      source_width,
      depth_engine_max_dim
    ));
    const int max_height = static_cast<int>(std::min<std::uint32_t>(
      source_height,
      depth_engine_max_dim
    ));
    const auto [width, height] = host_sbs_resolution_detail::aspect_aligned_dims(
      fitted_aspect,
      bounded_short_side,
      max_width,
      max_height
    );
    return {width, height};
  }

  /** True only for a tensor shape carried by the shipped V2 calibration. */
  inline constexpr bool host_sbs_v2_depth_shape_is_authenticated(
    const depth_tensor_shape_t shape
  ) noexcept {
    static_assert(!depth_coordinate_v2::model_calibrations.empty());
    return shape.valid() && depth_coordinate_v2::model_calibration_supports_shape(
                              depth_coordinate_v2::model_calibrations.front(),
                              static_cast<std::uint32_t>(shape.width),
                              static_cast<std::uint32_t>(shape.height)
                            );
  }

  /**
   * Runtime subtitle geometry for one authenticated DAV2 field.
   *
   * PP-OCR keeps a fixed tensor, while its boxes are projected into the current analysis field.
   * The safe detector rows are part of the generated OCR contract. Integer ceil projection keeps
   * host, shader, and dump bounds identical for landscape, ultrawide, and portrait fields.
   */
  struct subtitle_analysis_geometry_t {
    std::uint32_t field_width = 0u;
    std::uint32_t field_height = 0u;
    std::uint32_t roi_top = 0u;
    std::uint32_t roi_bottom = 0u;

    constexpr bool valid() const noexcept {
      return field_width > 0u && field_height > 0u &&
             roi_top < roi_bottom && roi_bottom <= field_height;
    }
  };

  inline constexpr subtitle_analysis_geometry_t fit_subtitle_analysis_geometry(
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const depth_tensor_shape_t field
  ) noexcept {
    if (!host_sbs_v2_depth_shape_is_authenticated(field)) {
      return {};
    }
    const auto roi = depth_coordinate_v2::subtitle_ocr_dynamic_roi(
      source_width,
      source_height,
      static_cast<std::uint32_t>(field.width),
      static_cast<std::uint32_t>(field.height)
    );
    if (!roi) {
      return {};
    }
    return {
      static_cast<std::uint32_t>(field.width),
      static_cast<std::uint32_t>(field.height),
      roi.top,
      roi.bottom,
    };
  }

  /** Height, in source pixels, of the exact bottom crop consumed by fixed-shape subtitle OCR. */
  inline constexpr std::uint32_t subtitle_ocr_source_crop_height(
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) noexcept {
    if (source_width == 0u || source_height == 0u) {
      return 0u;
    }
    const auto requested = depth_coordinate_v2::subtitle_ocr_ceil_div(
      static_cast<std::uint64_t>(source_width) *
        depth_coordinate_v2::subtitle_ocr_crop_aspect_height,
      depth_coordinate_v2::subtitle_ocr_crop_aspect_width
    );
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(requested, source_height));
  }

  /** Fit the exact source tensor requested by the fixed live Host SBS V2 calibration. */
  inline depth_tensor_shape_t fit_host_sbs_v2_depth_tensor_shape(
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) {
    return fit_depth_tensor_shape(
      source_width,
      source_height,
      config::host_sbs_v2_live_calibration::depth_short_side,
      static_cast<float>(config::host_sbs_v2_live_calibration::depth_max_aspect)
    );
  }

  /**
   * Preflight a source resolution through the exact production shape fitter and V2 allowlist.
   * This deliberately authenticates the resulting tensor, not an independently maintained list
   * of stream sizes. All twelve standard Moonlight 3D landscape/portrait choices map to the six
   * calibrated tensors; any custom source that fits another shape fails closed.
   */
  inline std::string_view host_sbs_v2_source_resolution_rejection_reason(
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) {
    if (source_width == 0u || source_height == 0u) {
      return "source extent is empty";
    }
    if (std::max(source_width, source_height) >
          host_sbs_v2_max_source_long_side ||
        static_cast<std::uint64_t>(source_width) * source_height >
          host_sbs_v2_max_source_pixels) {
      return "source raster exceeds the exact-area preprocessing budget";
    }
    if (!host_sbs_v2_depth_shape_is_authenticated(
          fit_host_sbs_v2_depth_tensor_shape(source_width, source_height)
        )) {
      return "fitted depth tensor is not authenticated";
    }
    return {};
  }

  inline bool host_sbs_v2_source_resolution_is_supported(
    const std::uint32_t source_width,
    const std::uint32_t source_height
  ) {
    return host_sbs_v2_source_resolution_rejection_reason(
             source_width,
             source_height
           ).empty();
  }

  /**
   * Select a video-only input for one already-active authenticated tensor shape.
   *
   * A non-fullscreen rectangle is kept only when it already has the exact tensor aspect. A near
   * miss may trim inward by at most 2% of its area and must then pass the ordinary fitter again.
   * This deliberately removes a thin player/frame border instead of admitting it to DAV2. It is
   * not a generic aspect converter: there is no stretching or padding, and a material mismatch
   * returns no plan (full-frame fallback). A full-capture rectangle is always ordinary full-frame
   * V2 and therefore returns no ROI plan.
   */
  inline std::optional<depth_video_region_plan_t> plan_host_sbs_v2_video_region(
    const depth_source_rect_t video_rect,
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const depth_tensor_shape_t required_shape,
    const float maximum_trim_fraction =
      host_sbs_v2_max_video_region_trim_fraction
  ) noexcept {
    if (!video_rect.valid() || video_rect.right > source_width ||
        video_rect.bottom > source_height ||
        !host_sbs_v2_depth_shape_is_authenticated(required_shape) ||
        !std::isfinite(maximum_trim_fraction) ||
        maximum_trim_fraction < 0.0f) {
      return std::nullopt;
    }
    // A semantic video that already covers the complete captured source is the ordinary
    // full-frame domain. Do not trim a few edge pixels merely because patch-size rounding makes
    // the model tensor's aspect differ slightly from the source raster.
    if (video_rect.left == 0u && video_rect.top == 0u &&
        video_rect.right == source_width && video_rect.bottom == source_height) {
      return std::nullopt;
    }
    const auto exact_shape = fit_host_sbs_v2_depth_tensor_shape(
      video_rect.width(),
      video_rect.height()
    );
    const std::uint64_t width = video_rect.width();
    const std::uint64_t height = video_rect.height();
    const std::uint64_t target_width = static_cast<std::uint32_t>(required_shape.width);
    const std::uint64_t target_height = static_cast<std::uint32_t>(required_shape.height);
    if (exact_shape == required_shape &&
        width * target_height == height * target_width &&
        host_sbs_v2_source_resolution_is_supported(
          video_rect.width(),
          video_rect.height()
        )) {
      return depth_video_region_plan_t {video_rect, exact_shape, 0.0f};
    }

    std::uint64_t fitted_width = width;
    std::uint64_t fitted_height = height;
    if (width * target_height > height * target_width) {
      fitted_width = height * target_width / target_height;
    } else if (width * target_height < height * target_width) {
      fitted_height = width * target_height / target_width;
    }
    if (fitted_width > width || fitted_height > height ||
        fitted_width < target_width || fitted_height < target_height) {
      return std::nullopt;
    }

    const std::uint64_t original_area = width * height;
    const std::uint64_t fitted_area = fitted_width * fitted_height;
    const float trimmed_fraction = static_cast<float>(
      1.0 - static_cast<double>(fitted_area) / static_cast<double>(original_area)
    );
    if (trimmed_fraction > maximum_trim_fraction) {
      return std::nullopt;
    }

    const auto remove_x = static_cast<std::uint32_t>(width - fitted_width);
    const auto remove_y = static_cast<std::uint32_t>(height - fitted_height);
    const auto left = video_rect.left + remove_x / 2u;
    const auto top = video_rect.top + remove_y / 2u;
    const depth_source_rect_t fitted_rect {
      left,
      top,
      left + static_cast<std::uint32_t>(fitted_width),
      top + static_cast<std::uint32_t>(fitted_height),
    };
    if (fitted_rect.left < video_rect.left || fitted_rect.top < video_rect.top ||
        fitted_rect.right > video_rect.right || fitted_rect.bottom > video_rect.bottom) {
      return std::nullopt;
    }
    const auto verified_shape = fit_host_sbs_v2_depth_tensor_shape(
      fitted_rect.width(),
      fitted_rect.height()
    );
    if (verified_shape != required_shape ||
        !host_sbs_v2_source_resolution_is_supported(
          fitted_rect.width(),
          fitted_rect.height()
        )) {
      return std::nullopt;
    }
    return depth_video_region_plan_t {
      fitted_rect,
      verified_shape,
      trimmed_fraction,
    };
  }

}  // namespace models

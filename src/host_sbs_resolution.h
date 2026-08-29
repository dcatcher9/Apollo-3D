#pragma once

#include "config.h"
#include "depth_coordinate_v2.h"
#include "model_manager.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

  /**
   * Centered integer half-open texel rectangle occupied by real source pixels in one fixed DAV2
   * tensor.
   * Pixels outside this rectangle are synthetic edge-replicated letterbox support and never
   * belong to the analysis domain.
   */
  struct depth_tensor_content_rect_t {
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

    constexpr bool valid(const depth_tensor_shape_t shape) const noexcept {
      return shape.valid() && valid() &&
             right <= static_cast<std::uint32_t>(shape.width) &&
             bottom <= static_cast<std::uint32_t>(shape.height);
    }

    constexpr bool full(const depth_tensor_shape_t shape) const noexcept {
      return shape.valid() && left == 0u && top == 0u &&
             right == static_cast<std::uint32_t>(shape.width) &&
             bottom == static_cast<std::uint32_t>(shape.height);
    }

    constexpr bool operator==(const depth_tensor_content_rect_t &) const = default;
  };

  struct depth_video_region_plan_t {
    depth_source_rect_t source_rect {};
    depth_tensor_shape_t tensor_shape {};
    depth_tensor_content_rect_t tensor_content {};
    float padded_area_fraction = 0.0f;

    constexpr bool operator==(const depth_video_region_plan_t &) const = default;
  };


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

  /** Exact public high-grid shape produced by the frozen convex-2x composite.
   *
   * The authenticated coarse DAV2 fit selects the profile; it is not a separately published
   * runtime domain. The fused engine exposes only the doubled high-grid input/output, and that
   * high output owns analysis, history, camera state, and publication. Rejecting non-production
   * coarse fits keeps this helper from becoming an unauthenticated general-purpose scaler.
   */
  inline constexpr std::optional<depth_tensor_shape_t>
  host_sbs_convex2x_field_shape(const depth_tensor_shape_t coarse) noexcept {
    if (!host_sbs_v2_depth_shape_is_authenticated(coarse) ||
        coarse.width > std::numeric_limits<int>::max() / 2 ||
        coarse.height > std::numeric_limits<int>::max() / 2) {
      return std::nullopt;
    }
    return depth_tensor_shape_t {2 * coarse.width, 2 * coarse.height};
  }

  inline constexpr bool host_sbs_convex2x_shape_relation(
    const depth_tensor_shape_t coarse,
    const depth_tensor_shape_t refined
  ) noexcept {
    const auto expected = host_sbs_convex2x_field_shape(coarse);
    return expected && *expected == refined;
  }

  /** Operator/engine shape support only; this does not grant live-rendering authority. */
  inline constexpr bool host_sbs_convex2x_operator_shape_is_supported(
    const depth_tensor_shape_t coarse
  ) noexcept {
    return host_sbs_convex2x_field_shape(coarse).has_value();
  }

  /** Double an authenticated coarse content rectangle without a second aspect-ratio fit.
   *
   * Re-fitting can move an odd content edge and therefore change the 2x pixel-shuffle phase.
   * Scaling every half-open edge preserves the exact source/content correspondence.
   */
  inline constexpr std::optional<depth_tensor_content_rect_t>
  host_sbs_convex2x_content_rect(
    const depth_tensor_content_rect_t coarse_content,
    const depth_tensor_shape_t coarse_shape
  ) noexcept {
    if (!coarse_content.valid(coarse_shape) ||
        !host_sbs_convex2x_field_shape(coarse_shape)) {
      return std::nullopt;
    }
    return depth_tensor_content_rect_t {
      2u * coarse_content.left,
      2u * coarse_content.top,
      2u * coarse_content.right,
      2u * coarse_content.bottom,
    };
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
    std::uint32_t ribbon_min_bottom = 0u;
    depth_tensor_content_rect_t tensor_content {};

    constexpr bool valid() const noexcept {
      return field_width > 0u && field_height > 0u &&
             roi_top < roi_bottom && roi_bottom <= field_height &&
             roi_top < ribbon_min_bottom && ribbon_min_bottom <= roi_bottom &&
             tensor_content.valid({
               static_cast<int>(field_width),
               static_cast<int>(field_height),
             }) &&
             roi_top >= tensor_content.top && roi_bottom <= tensor_content.bottom;
    }
  };

  inline constexpr subtitle_analysis_geometry_t fit_subtitle_analysis_geometry(
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const depth_tensor_shape_t field,
    const depth_tensor_content_rect_t tensor_content
  ) noexcept {
    if (!field.valid() ||
        !depth_coordinate_v2::subtitle_ocr_field_is_calibrated(
          static_cast<std::uint32_t>(field.width),
          static_cast<std::uint32_t>(field.height)
        ) ||
        !tensor_content.valid(field) || source_width == 0u || source_height == 0u) {
      return {};
    }
    const auto crop_height = std::min<std::uint64_t>(
      source_height,
      depth_coordinate_v2::subtitle_ocr_ceil_div(
        static_cast<std::uint64_t>(source_width) *
          depth_coordinate_v2::subtitle_ocr_crop_aspect_height,
        depth_coordinate_v2::subtitle_ocr_crop_aspect_width
      )
    );
    const auto crop_top = static_cast<std::uint64_t>(source_height) - crop_height;
    const auto denominator = static_cast<std::uint64_t>(source_height) *
      depth_coordinate_v2::subtitle_ocr_output_height;
    const auto project_row = [&](const std::uint32_t detector_y) constexpr {
      const auto source_numerator = crop_top *
        depth_coordinate_v2::subtitle_ocr_output_height +
        static_cast<std::uint64_t>(detector_y) * crop_height;
      return static_cast<std::uint32_t>(tensor_content.top +
        std::min<std::uint64_t>(
          depth_coordinate_v2::subtitle_ocr_ceil_div(
            source_numerator * tensor_content.height(), denominator
          ),
          tensor_content.height()
        ));
    };
    const auto roi_top = project_row(depth_coordinate_v2::subtitle_ocr_safe_row_top);
    const auto roi_bottom = project_row(depth_coordinate_v2::subtitle_ocr_safe_row_bottom);
    const auto ribbon_min_bottom = project_row(
      depth_coordinate_v2::subtitle_ocr_safe_row_bottom -
        depth_coordinate_v2::subtitle_ocr_ribbon_bottom_tolerance_pixels
    );
    if (roi_top >= roi_bottom || ribbon_min_bottom <= roi_top ||
        ribbon_min_bottom > roi_bottom) {
      return {};
    }
    return {
      static_cast<std::uint32_t>(field.width),
      static_cast<std::uint32_t>(field.height),
      roi_top,
      roi_bottom,
      ribbon_min_bottom,
      tensor_content,
    };
  }

  inline constexpr subtitle_analysis_geometry_t fit_subtitle_analysis_geometry(
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const depth_tensor_shape_t field
  ) noexcept {
    return fit_subtitle_analysis_geometry(
      source_width,
      source_height,
      field,
      {
        0u,
        0u,
        static_cast<std::uint32_t>(std::max(field.width, 0)),
        static_cast<std::uint32_t>(std::max(field.height, 0)),
      }
    );
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
   * Select an exact window-region source for one already-active authenticated tensor shape.
   *
   * The source rectangle is never cropped or stretched. A centered integer half-open content
   * rectangle letterboxes it into the fixed tensor, and the preprocess edge-replicates pixels
   * outside that rectangle. A full-capture rectangle remains ordinary full-frame V2 and returns
   * no ROI plan.
   */
  inline std::optional<depth_video_region_plan_t> plan_host_sbs_v2_video_region(
    const depth_source_rect_t video_rect,
    const std::uint32_t source_width,
    const std::uint32_t source_height,
    const depth_tensor_shape_t required_shape
  ) noexcept {
    if (!video_rect.valid() || video_rect.right > source_width ||
        video_rect.bottom > source_height ||
        !host_sbs_v2_depth_shape_is_authenticated(required_shape)) {
      return std::nullopt;
    }
    // A semantic video that already covers the complete captured source is the ordinary
    // full-frame domain. Do not trim a few edge pixels merely because patch-size rounding makes
    // the model tensor's aspect differ slightly from the source raster.
    if (video_rect.left == 0u && video_rect.top == 0u &&
        video_rect.right == source_width && video_rect.bottom == source_height) {
      return std::nullopt;
    }

    const std::uint64_t width = video_rect.width();
    const std::uint64_t height = video_rect.height();
    const std::uint64_t target_width = static_cast<std::uint32_t>(required_shape.width);
    const std::uint64_t target_height = static_cast<std::uint32_t>(required_shape.height);
    depth_tensor_content_rect_t content {
      0u,
      0u,
      static_cast<std::uint32_t>(required_shape.width),
      static_cast<std::uint32_t>(required_shape.height),
    };
    if (width * target_height > height * target_width) {
      const auto content_height = static_cast<std::uint32_t>(std::max<std::uint64_t>(
        1u,
        target_width * height / width
      ));
      content.top = (static_cast<std::uint32_t>(required_shape.height) - content_height) / 2u;
      content.bottom = content.top + content_height;
    } else if (width * target_height < height * target_width) {
      const auto content_width = static_cast<std::uint32_t>(std::max<std::uint64_t>(
        1u,
        target_height * width / height
      ));
      content.left = (static_cast<std::uint32_t>(required_shape.width) - content_width) / 2u;
      content.right = content.left + content_width;
    }
    if (!content.valid(required_shape)) {
      return std::nullopt;
    }
    const double content_area = static_cast<double>(content.width()) * content.height();
    const double tensor_area = static_cast<double>(target_width) * target_height;
    const float padded_fraction = static_cast<float>(1.0 - content_area / tensor_area);
    return depth_video_region_plan_t {
      video_rect,
      required_shape,
      content,
      padded_fraction,
    };
  }

}  // namespace models

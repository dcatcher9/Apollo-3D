/**
 * @file src/sbs_roi_feature_detector.cpp
 * @brief Bounded feature-grid candidate extraction for Host SBS content ROIs.
 */
#include "sbs_roi_feature_detector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace sbs_roi {
  namespace {
    constexpr std::size_t tracker_candidate_budget = 64;

    struct cell_rect_t {
      std::uint16_t left = 0;
      std::uint16_t top = 0;
      std::uint16_t right = 0;
      std::uint16_t bottom = 0;
    };

    struct component_t {
      cell_rect_t bounds;
      std::uint32_t label = 0;
      std::size_t cells = 0;
    };

    struct component_set_t {
      std::vector<component_t> components;
      std::vector<std::uint32_t> labels;
      bool overflow = false;
    };

    struct cell_gutter_t {
      cell_rect_t bounds;
      float confidence = 0.0f;
    };

    struct cell_column_t {
      cell_rect_t bounds;
      std::uint64_t evidence = 0;
    };

    bool finite(float value) {
      return std::isfinite(value);
    }

    bool valid_rect(const normalized_rect_t &rect) {
      return finite(rect.left) &&
             finite(rect.top) &&
             finite(rect.right) &&
             finite(rect.bottom) &&
             rect.left >= 0.0f &&
             rect.top >= 0.0f &&
             rect.right <= 1.0f &&
             rect.bottom <= 1.0f &&
             rect.left < rect.right &&
             rect.top < rect.bottom;
    }

    bool valid_point(const normalized_point_t &point) {
      return finite(point.x) &&
             finite(point.y) &&
             point.x >= 0.0f &&
             point.x < 1.0f &&
             point.y >= 0.0f &&
             point.y < 1.0f;
    }

    bool valid_fraction(float value) {
      return finite(value) && value >= 0.0f && value <= 1.0f;
    }

    std::size_t cell_index(
      std::uint16_t x,
      std::uint16_t y,
      std::uint16_t width
    ) {
      return static_cast<std::size_t>(y) * width + x;
    }

    std::uint16_t scaled_cells(
      std::uint16_t nominal_cells,
      std::uint16_t actual_extent,
      std::uint16_t nominal_extent,
      bool allow_zero = false
    ) {
      if (nominal_cells == 0) {
        return 0;
      }
      const auto scaled =
        (static_cast<std::uint32_t>(nominal_cells) * actual_extent +
         nominal_extent / 2u) /
        nominal_extent;
      return static_cast<std::uint16_t>(
        std::max<std::uint32_t>(allow_zero ? 0u : 1u, scaled)
      );
    }

    normalized_rect_t normalize_rect(
      const cell_rect_t &rect,
      std::uint16_t width,
      std::uint16_t height
    ) {
      return {
        static_cast<float>(rect.left) / width,
        static_cast<float>(rect.top) / height,
        static_cast<float>(rect.right) / width,
        static_cast<float>(rect.bottom) / height,
      };
    }

    float normalized_area(const cell_rect_t &rect, std::uint16_t width, std::uint16_t height) {
      return static_cast<float>(rect.right - rect.left) *
             static_cast<float>(rect.bottom - rect.top) /
             (static_cast<float>(width) * height);
    }

    bool contains(const normalized_rect_t &rect, const normalized_point_t &point) {
      return point.x >= rect.left &&
             point.x < rect.right &&
             point.y >= rect.top &&
             point.y < rect.bottom;
    }

    bool contains_cell(
      const normalized_rect_t &rect,
      std::uint16_t x,
      std::uint16_t y,
      std::uint16_t width,
      std::uint16_t height
    ) {
      const auto center_x = (static_cast<float>(x) + 0.5f) / width;
      const auto center_y = (static_cast<float>(y) + 0.5f) / height;
      return center_x >= rect.left &&
             center_x < rect.right &&
             center_y >= rect.top &&
             center_y < rect.bottom;
    }

    float intersection_support(
      const cell_rect_t &rect,
      const cell_rect_t &column
    ) {
      const auto left = std::max(rect.left, column.left);
      const auto right = std::min(rect.right, column.right);
      if (right <= left) {
        return 0.0f;
      }
      return static_cast<float>(right - left) /
             static_cast<float>(rect.right - rect.left);
    }

    bool crosses_barrier(
      const cell_rect_t &rect,
      const std::vector<std::uint8_t> &barriers
    ) {
      if (rect.right <= rect.left + 1) {
        return false;
      }
      for (auto x = static_cast<std::uint16_t>(rect.left + 1); x < rect.right; ++x) {
        if (barriers[x] != 0) {
          return true;
        }
      }
      return false;
    }

    std::vector<std::uint8_t> bridge_small_gaps(
      const std::vector<std::uint8_t> &mask,
      std::uint16_t width,
      std::uint16_t height,
      std::uint16_t bridge_x,
      std::uint16_t bridge_y,
      const std::vector<std::uint8_t> &barriers
    ) {
      const auto horizontal_pass =
        [&](const std::vector<std::uint8_t> &input) {
          auto output = input;
          if (bridge_x == 0) {
            return output;
          }
          for (std::uint16_t y = 0; y < height; ++y) {
            std::optional<std::uint16_t> previous;
            for (std::uint16_t x = 0; x < width; ++x) {
              if (barriers[x] != 0) {
                previous.reset();
                continue;
              }
              if (input[cell_index(x, y, width)] == 0) {
                continue;
              }
              if (previous && x > *previous + 1 && x - *previous - 1 <= bridge_x) {
                for (auto fill_x = static_cast<std::uint16_t>(*previous + 1); fill_x < x; ++fill_x) {
                  output[cell_index(fill_x, y, width)] = 1;
                }
              }
              previous = x;
            }
          }
          return output;
        };
      const auto vertical_pass =
        [&](const std::vector<std::uint8_t> &input) {
          auto output = input;
          if (bridge_y == 0) {
            return output;
          }
          for (std::uint16_t x = 0; x < width; ++x) {
            if (barriers[x] != 0) {
              continue;
            }
            std::optional<std::uint16_t> previous;
            for (std::uint16_t y = 0; y < height; ++y) {
              if (input[cell_index(x, y, width)] == 0) {
                continue;
              }
              if (previous && y > *previous + 1 && y - *previous - 1 <= bridge_y) {
                for (auto fill_y = static_cast<std::uint16_t>(*previous + 1); fill_y < y; ++fill_y) {
                  output[cell_index(x, fill_y, width)] = 1;
                }
              }
              previous = y;
            }
          }
          return output;
        };

      auto bridged = horizontal_pass(mask);
      bridged = vertical_pass(bridged);
      return horizontal_pass(bridged);
    }

    std::vector<std::uint8_t> open_video_support(
      const std::vector<std::uint8_t> &support,
      const std::vector<std::uint8_t> &activity_seeds,
      std::uint16_t width,
      std::uint16_t height,
      const std::vector<std::uint8_t> &barriers
    ) {
      std::vector<std::uint8_t> eroded(support.size(), 0);
      for (std::uint16_t y = 1; y + 1 < height; ++y) {
        for (std::uint16_t x = 1; x + 1 < width; ++x) {
          if (barriers[x] != 0) {
            continue;
          }
          bool full_neighborhood = true;
          for (int offset_y = -1; offset_y <= 1 && full_neighborhood; ++offset_y) {
            for (int offset_x = -1; offset_x <= 1; ++offset_x) {
              const auto neighbor_x =
                static_cast<std::uint16_t>(
                  static_cast<int>(x) + offset_x
                );
              const auto neighbor_y =
                static_cast<std::uint16_t>(
                  static_cast<int>(y) + offset_y
                );
              if (barriers[neighbor_x] != 0 || support[cell_index(neighbor_x, neighbor_y, width)] == 0) {
                full_neighborhood = false;
                break;
              }
            }
          }
          eroded[cell_index(x, y, width)] = full_neighborhood ? 1 : 0;
        }
      }

      std::vector<std::uint8_t> opened(support.size(), 0);
      for (std::uint16_t y = 0; y < height; ++y) {
        for (std::uint16_t x = 0; x < width; ++x) {
          const auto index = cell_index(x, y, width);
          if (barriers[x] != 0 || support[index] == 0) {
            continue;
          }
          const auto min_x = x == 0 ? 0 : x - 1;
          const auto max_x =
            static_cast<std::uint16_t>(
              std::min<std::uint32_t>(width - 1, x + 1)
            );
          const auto min_y = y == 0 ? 0 : y - 1;
          const auto max_y =
            static_cast<std::uint16_t>(
              std::min<std::uint32_t>(height - 1, y + 1)
            );
          for (auto neighbor_y = min_y; neighbor_y <= max_y && opened[index] == 0; ++neighbor_y) {
            for (auto neighbor_x = min_x; neighbor_x <= max_x; ++neighbor_x) {
              if (eroded[cell_index(neighbor_x, neighbor_y, width)] != 0) {
                opened[index] = 1;
                break;
              }
            }
          }
          if (activity_seeds[index] != 0) {
            opened[index] = 1;
          }
        }
      }
      return opened;
    }

    component_set_t collect_components(
      const std::vector<std::uint8_t> &mask,
      std::uint16_t width,
      std::uint16_t height,
      std::size_t max_components
    ) {
      component_set_t result;
      result.labels.assign(mask.size(), 0);
      result.components.reserve(
        std::min(max_components, mask.size())
      );
      std::vector<std::size_t> queue;
      queue.reserve(mask.size());
      std::uint32_t next_label = 1;

      for (std::uint16_t seed_y = 0; seed_y < height; ++seed_y) {
        for (std::uint16_t seed_x = 0; seed_x < width; ++seed_x) {
          const auto seed = cell_index(seed_x, seed_y, width);
          if (mask[seed] == 0 || result.labels[seed] != 0) {
            continue;
          }
          if (result.components.size() >= max_components) {
            result.components.clear();
            result.labels.clear();
            result.overflow = true;
            return result;
          }

          component_t component;
          component.bounds = {
            seed_x,
            seed_y,
            static_cast<std::uint16_t>(seed_x + 1),
            static_cast<std::uint16_t>(seed_y + 1),
          };
          component.label = next_label;
          queue.clear();
          queue.push_back(seed);
          result.labels[seed] = next_label;

          for (std::size_t head = 0; head < queue.size(); ++head) {
            const auto current = queue[head];
            const auto current_x =
              static_cast<std::uint16_t>(current % width);
            const auto current_y =
              static_cast<std::uint16_t>(current / width);
            ++component.cells;
            component.bounds.left =
              std::min(component.bounds.left, current_x);
            component.bounds.top =
              std::min(component.bounds.top, current_y);
            component.bounds.right =
              std::max(
                component.bounds.right,
                static_cast<std::uint16_t>(current_x + 1)
              );
            component.bounds.bottom =
              std::max(
                component.bounds.bottom,
                static_cast<std::uint16_t>(current_y + 1)
              );

            const std::array<std::pair<int, int>, 4> neighbors {{
              {-1, 0},
              {1, 0},
              {0, -1},
              {0, 1},
            }};
            for (const auto &[offset_x, offset_y] : neighbors) {
              const auto neighbor_x =
                static_cast<int>(current_x) + offset_x;
              const auto neighbor_y =
                static_cast<int>(current_y) + offset_y;
              if (neighbor_x < 0 || neighbor_x >= width || neighbor_y < 0 || neighbor_y >= height) {
                continue;
              }
              const auto neighbor = cell_index(
                static_cast<std::uint16_t>(neighbor_x),
                static_cast<std::uint16_t>(neighbor_y),
                width
              );
              if (mask[neighbor] == 0 || result.labels[neighbor] != 0) {
                continue;
              }
              result.labels[neighbor] = next_label;
              queue.push_back(neighbor);
            }
          }

          result.components.push_back(component);
          ++next_label;
        }
      }
      return result;
    }

    std::uint64_t column_evidence(
      feature_grid_view_t grid,
      std::uint16_t x
    ) {
      std::uint64_t evidence = 0;
      for (std::uint16_t y = 0; y < grid.height; ++y) {
        const auto &cell = grid.cells[cell_index(x, y, grid.width)];
        evidence +=
          std::max(cell.temporal_occupancy_q8, cell.photographic_density_q8);
      }
      return evidence;
    }

    std::vector<cell_gutter_t> find_structural_gutters(
      feature_grid_view_t grid,
      const feature_detector_config_t &config
    ) {
      const auto body_top = static_cast<std::uint16_t>(grid.height / 12u);
      const auto body_bottom =
        static_cast<std::uint16_t>(grid.height - grid.height / 16u);
      const auto body_height = std::max<std::uint16_t>(1, body_bottom - body_top);
      std::vector<std::uint8_t> stable_columns(grid.width, 0);
      std::vector<float> confidence(grid.width, 0.0f);
      std::vector<std::uint64_t> evidence(grid.width, 0);

      for (std::uint16_t x = 0; x < grid.width; ++x) {
        std::uint32_t stable_rows = 0;
        std::uint64_t stability_sum = 0;
        for (auto y = body_top; y < body_bottom; ++y) {
          const auto stability =
            grid.cells[cell_index(x, y, grid.width)].gutter_stability_q8;
          stability_sum += stability;
          if (stability >= config.gutter_stability_threshold_q8) {
            ++stable_rows;
          }
        }
        confidence[x] =
          static_cast<float>(stability_sum) /
          (static_cast<float>(body_height) * 255.0f);
        stable_columns[x] =
          static_cast<float>(stable_rows) / body_height >=
              config.gutter_row_fraction ?
            1 :
            0;
        evidence[x] = column_evidence(grid, x);
      }

      const auto min_width = scaled_cells(
        config.min_gutter_width_cells,
        grid.width,
        nominal_feature_grid_width
      );
      std::vector<cell_gutter_t> gutters;
      std::uint16_t x = 0;
      while (x < grid.width) {
        if (stable_columns[x] == 0) {
          ++x;
          continue;
        }
        const auto left = x;
        float confidence_sum = 0.0f;
        while (x < grid.width && stable_columns[x] != 0) {
          confidence_sum += confidence[x];
          ++x;
        }
        const auto right = x;
        if (right - left < min_width) {
          continue;
        }

        const auto left_evidence =
          std::accumulate(evidence.begin(), evidence.begin() + left, std::uint64_t {0});
        const auto right_evidence =
          std::accumulate(evidence.begin() + right, evidence.end(), std::uint64_t {0});
        const auto minimum_side_evidence =
          static_cast<std::uint64_t>(config.video_photo_threshold_q8) *
          grid.height *
          scaled_cells(
            config.min_component_width_cells,
            grid.width,
            nominal_feature_grid_width
          );
        if (left_evidence < minimum_side_evidence || right_evidence < minimum_side_evidence) {
          continue;
        }

        gutters.push_back({
          {left, 0, right, grid.height},
          confidence_sum / (right - left),
        });
      }
      return gutters;
    }

    std::vector<std::uint8_t> barrier_columns(
      std::uint16_t width,
      const std::vector<cell_gutter_t> &gutters
    ) {
      std::vector<std::uint8_t> barriers(width, 0);
      for (const auto &gutter : gutters) {
        for (auto x = gutter.bounds.left; x < gutter.bounds.right; ++x) {
          barriers[x] = 1;
        }
      }
      return barriers;
    }

    std::vector<cell_column_t> find_columns(
      feature_grid_view_t grid,
      const std::vector<cell_gutter_t> &gutters
    ) {
      std::vector<cell_column_t> columns;
      std::uint16_t left = 0;
      const auto append = [&](std::uint16_t right) {
        if (right <= left) {
          return;
        }
        std::uint64_t evidence = 0;
        for (auto x = left; x < right; ++x) {
          evidence += column_evidence(grid, x);
        }
        if (evidence > 0) {
          columns.push_back({{left, 0, right, grid.height}, evidence});
        }
      };

      for (const auto &gutter : gutters) {
        append(gutter.bounds.left);
        left = gutter.bounds.right;
      }
      append(grid.width);
      return columns;
    }

    std::optional<std::size_t> choose_primary_column(
      const std::vector<cell_column_t> &columns,
      float width_ratio,
      bool &ambiguous
    ) {
      ambiguous = false;
      if (columns.empty()) {
        return std::nullopt;
      }

      std::vector<std::size_t> order(columns.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(
        order.begin(),
        order.end(),
        [&](std::size_t lhs, std::size_t rhs) {
          const auto lhs_width =
            columns[lhs].bounds.right - columns[lhs].bounds.left;
          const auto rhs_width =
            columns[rhs].bounds.right - columns[rhs].bounds.left;
          return std::tie(lhs_width, columns[lhs].evidence) >
                 std::tie(rhs_width, columns[rhs].evidence);
        }
      );

      if (order.size() == 1) {
        return order.front();
      }
      const auto first_width =
        columns[order[0]].bounds.right - columns[order[0]].bounds.left;
      const auto second_width =
        columns[order[1]].bounds.right - columns[order[1]].bounds.left;
      if (static_cast<float>(first_width) < static_cast<float>(second_width) * width_ratio) {
        ambiguous = true;
        return std::nullopt;
      }
      return order.front();
    }

    float average_feature(
      feature_grid_view_t grid,
      const cell_rect_t &rect,
      bool activity
    ) {
      std::uint64_t sum = 0;
      for (auto y = rect.top; y < rect.bottom; ++y) {
        for (auto x = rect.left; x < rect.right; ++x) {
          const auto &cell = grid.cells[cell_index(x, y, grid.width)];
          sum += activity ?
                   cell.temporal_occupancy_q8 :
                   cell.photographic_density_q8;
        }
      }
      const auto cell_count =
        static_cast<std::uint64_t>(rect.right - rect.left) *
        (rect.bottom - rect.top);
      return cell_count == 0 ?
               0.0f :
               static_cast<float>(sum) /
                 (static_cast<float>(cell_count) * 255.0f);
    }

    std::uint16_t weighted_lower_bound(
      const std::vector<std::uint64_t> &weights,
      double target
    ) {
      double cumulative = 0.0;
      for (std::uint16_t index = 0; index < weights.size(); ++index) {
        cumulative += static_cast<double>(weights[index]);
        if (cumulative >= target) {
          return index;
        }
      }
      return static_cast<std::uint16_t>(weights.size() - 1);
    }

    std::uint16_t weighted_upper_offset(
      const std::vector<std::uint64_t> &weights,
      double target
    ) {
      double cumulative = 0.0;
      for (std::uint16_t offset = 0; offset < weights.size(); ++offset) {
        cumulative += static_cast<double>(
          weights[weights.size() - 1 - offset]
        );
        if (cumulative >= target) {
          return offset;
        }
      }
      return static_cast<std::uint16_t>(weights.size() - 1);
    }

    std::uint64_t component_photo_weight(
      const component_t &component,
      const component_set_t &components,
      feature_grid_view_t grid,
      const cell_rect_t &rect
    ) {
      std::uint64_t weight = 0;
      for (auto y = rect.top; y < rect.bottom; ++y) {
        for (auto x = rect.left; x < rect.right; ++x) {
          const auto index = cell_index(x, y, grid.width);
          if (components.labels[index] == component.label) {
            weight += grid.cells[index].photographic_density_q8;
          }
        }
      }
      return weight;
    }

    cell_rect_t trim_content_bounds(
      const component_t &component,
      const component_set_t &components,
      feature_grid_view_t grid,
      float trim_fraction,
      float minimum_density_gain
    ) {
      const auto width = component.bounds.right - component.bounds.left;
      const auto height = component.bounds.bottom - component.bounds.top;
      std::vector<std::uint64_t> x_weights(width, 0);
      std::vector<std::uint64_t> y_weights(height, 0);
      std::uint64_t total = 0;

      for (auto y = component.bounds.top; y < component.bounds.bottom; ++y) {
        for (auto x = component.bounds.left; x < component.bounds.right; ++x) {
          const auto index = cell_index(x, y, grid.width);
          if (components.labels[index] != component.label) {
            continue;
          }
          const auto weight =
            grid.cells[index].photographic_density_q8;
          x_weights[x - component.bounds.left] += weight;
          y_weights[y - component.bounds.top] += weight;
          total += weight;
        }
      }
      if (total == 0) {
        return component.bounds;
      }

      // Each axis removes at most half of the total trim budget so their
      // rectangular intersection retains at least approximately 1-trim.
      const auto tail =
        static_cast<double>(total) * trim_fraction * 0.25;
      const auto lower_x = weighted_lower_bound(x_weights, tail);
      const auto lower_y = weighted_lower_bound(y_weights, tail);
      const auto upper_x_from_end =
        weighted_upper_offset(x_weights, tail);
      const auto upper_y_from_end =
        weighted_upper_offset(y_weights, tail);
      const cell_rect_t trimmed {
        static_cast<std::uint16_t>(component.bounds.left + lower_x),
        static_cast<std::uint16_t>(component.bounds.top + lower_y),
        static_cast<std::uint16_t>(
          component.bounds.right - upper_x_from_end
        ),
        static_cast<std::uint16_t>(
          component.bounds.bottom - upper_y_from_end
        ),
      };
      if (trimmed.left >= trimmed.right || trimmed.top >= trimmed.bottom) {
        return component.bounds;
      }

      const auto retained =
        component_photo_weight(component, components, grid, trimmed);
      const auto raw_area =
        static_cast<double>(component.bounds.right - component.bounds.left) *
        (component.bounds.bottom - component.bounds.top);
      const auto trimmed_area =
        static_cast<double>(trimmed.right - trimmed.left) *
        (trimmed.bottom - trimmed.top);
      const auto coverage =
        static_cast<double>(retained) / total;
      const auto raw_density = static_cast<double>(total) / raw_area;
      const auto trimmed_density =
        static_cast<double>(retained) / trimmed_area;
      return coverage >= 1.0 - trim_fraction &&
                 trimmed_density >= raw_density * minimum_density_gain ?
               trimmed :
               component.bounds;
    }

    cell_rect_t add_content_halo(
      cell_rect_t rect,
      std::uint16_t halo_x,
      std::uint16_t halo_y,
      std::uint16_t width,
      std::uint16_t height,
      const std::vector<std::uint8_t> &barriers
    ) {
      for (std::uint16_t step = 0; step < halo_x; ++step) {
        if (rect.left > 0 && barriers[rect.left - 1] == 0) {
          --rect.left;
        }
        if (rect.right < width && barriers[rect.right] == 0) {
          ++rect.right;
        }
      }
      rect.top =
        rect.top > halo_y ?
          static_cast<std::uint16_t>(rect.top - halo_y) :
          0;
      rect.bottom =
        static_cast<std::uint16_t>(
          std::min<std::uint32_t>(height, rect.bottom + halo_y)
        );
      return rect;
    }

    float gutter_confidence_for(
      const cell_rect_t &rect,
      const std::vector<cell_gutter_t> &gutters
    ) {
      if (gutters.empty()) {
        return 1.0f;
      }
      std::optional<std::pair<std::uint16_t, float>> nearest_left;
      std::optional<std::pair<std::uint16_t, float>> nearest_right;
      for (const auto &gutter : gutters) {
        if (gutter.bounds.right <= rect.left) {
          const auto distance =
            static_cast<std::uint16_t>(rect.left - gutter.bounds.right);
          if (!nearest_left || distance < nearest_left->first) {
            nearest_left = {distance, gutter.confidence};
          }
        } else if (gutter.bounds.left >= rect.right) {
          const auto distance =
            static_cast<std::uint16_t>(gutter.bounds.left - rect.right);
          if (!nearest_right || distance < nearest_right->first) {
            nearest_right = {distance, gutter.confidence};
          }
        }
      }
      if (nearest_left && nearest_right) {
        return std::min(nearest_left->second, nearest_right->second);
      }
      if (nearest_left) {
        return nearest_left->second;
      }
      if (nearest_right) {
        return nearest_right->second;
      }
      return 0.0f;
    }

    candidate_t make_candidate(
      roi_kind_e kind,
      const cell_rect_t &rect,
      feature_grid_view_t grid,
      const std::vector<cell_gutter_t> &gutters,
      const std::vector<std::uint8_t> &barriers,
      const std::optional<cell_rect_t> &primary_column,
      bool primary_ambiguous,
      const std::optional<normalized_point_t> &interaction
    ) {
      candidate_t candidate;
      candidate.kind = kind;
      candidate.rect = normalize_rect(rect, grid.width, grid.height);
      candidate.temporal_occupancy =
        average_feature(grid, rect, true);
      candidate.photographic_density =
        average_feature(grid, rect, false);
      candidate.crosses_stable_gutter =
        crosses_barrier(rect, barriers);
      candidate.gutter_confidence =
        gutter_confidence_for(rect, gutters);

      if (primary_ambiguous) {
        candidate.primary_column_support = 0.0f;
        candidate.inside_primary_column = false;
      } else if (!primary_column) {
        candidate.primary_column_support = 1.0f;
        candidate.inside_primary_column = true;
      } else {
        candidate.primary_column_support =
          intersection_support(rect, *primary_column);
        candidate.inside_primary_column =
          candidate.primary_column_support >= 0.98f;
      }
      candidate.recent_interaction =
        interaction && contains(candidate.rect, *interaction);
      return candidate;
    }

    std::size_t component_mask_cells(
      const component_t &component,
      const component_set_t &components,
      const std::vector<std::uint8_t> &mask,
      std::uint16_t grid_width
    ) {
      std::size_t count = 0;
      for (auto y = component.bounds.top; y < component.bounds.bottom; ++y) {
        for (auto x = component.bounds.left; x < component.bounds.right; ++x) {
          const auto index = cell_index(x, y, grid_width);
          if (components.labels[index] == component.label && mask[index] != 0) {
            ++count;
          }
        }
      }
      return count;
    }

    bool content_component_eligible(
      const component_t &component,
      const component_set_t &components,
      const std::vector<std::uint8_t> &original_mask,
      feature_grid_view_t grid,
      const feature_detector_config_t &config
    ) {
      const auto width =
        component.bounds.right - component.bounds.left;
      const auto height =
        component.bounds.bottom - component.bounds.top;
      const auto width_fraction =
        static_cast<float>(width) / grid.width;
      const auto height_fraction =
        static_cast<float>(height) / grid.height;
      const auto area_fraction = width_fraction * height_fraction;
      const auto aspect =
        std::max(
          width_fraction / std::max(height_fraction, 0.0001f),
          height_fraction / std::max(width_fraction, 0.0001f)
        );
      const auto bounding_cells =
        static_cast<std::size_t>(width) * height;
      const auto filled_cells = component_mask_cells(
        component,
        components,
        original_mask,
        grid.width
      );
      const auto fill =
        bounding_cells == 0 ?
          0.0f :
          static_cast<float>(filled_cells) / bounding_cells;
      std::size_t occupied_rows = 0;
      for (auto y = component.bounds.top; y < component.bounds.bottom; ++y) {
        std::size_t row_cells = 0;
        for (auto x = component.bounds.left; x < component.bounds.right; ++x) {
          const auto index = cell_index(x, y, grid.width);
          row_cells +=
            components.labels[index] == component.label &&
                original_mask[index] != 0 ?
              1 :
              0;
        }
        occupied_rows +=
          static_cast<float>(row_cells) / width >=
              config.content_axis_fill ?
            1 :
            0;
      }
      std::size_t occupied_columns = 0;
      for (auto x = component.bounds.left; x < component.bounds.right; ++x) {
        std::size_t column_cells = 0;
        for (auto y = component.bounds.top; y < component.bounds.bottom; ++y) {
          const auto index = cell_index(x, y, grid.width);
          column_cells +=
            components.labels[index] == component.label &&
                original_mask[index] != 0 ?
              1 :
              0;
        }
        occupied_columns +=
          static_cast<float>(column_cells) / height >=
              config.content_axis_fill ?
            1 :
            0;
      }
      const auto row_occupancy =
        static_cast<float>(occupied_rows) / height;
      const auto column_occupancy =
        static_cast<float>(occupied_columns) / width;
      return area_fraction >= config.content_min_area &&
             width_fraction >= config.content_min_width &&
             height_fraction >= config.content_min_height &&
             aspect <= config.content_max_aspect_ratio &&
             fill >= config.content_min_fill &&
             row_occupancy >= config.content_min_axis_occupancy &&
             column_occupancy >= config.content_min_axis_occupancy;
    }

    float candidate_area(const candidate_t &candidate) {
      return (candidate.rect.right - candidate.rect.left) *
             (candidate.rect.bottom - candidate.rect.top);
    }

    float candidate_intersection_area(
      const candidate_t &lhs,
      const candidate_t &rhs
    ) {
      return std::max(
               0.0f,
               std::min(lhs.rect.right, rhs.rect.right) -
                 std::max(lhs.rect.left, rhs.rect.left)
             ) *
             std::max(
               0.0f,
               std::min(lhs.rect.bottom, rhs.rect.bottom) -
                 std::max(lhs.rect.top, rhs.rect.top)
             );
    }

    bool near_identity(const candidate_t &candidate) {
      return candidate_area(candidate) >= 0.90f &&
             candidate.rect.left <= 0.05f &&
             candidate.rect.top <= 0.05f &&
             candidate.rect.right >= 0.95f &&
             candidate.rect.bottom >= 0.95f;
    }

    void fail_closed_without_structural_separation(
      feature_detection_result_t &result,
      const feature_detector_config_t &config
    ) {
      if (!result.gutters.empty() || result.primary_column_ambiguous) {
        return;
      }
      std::vector<const candidate_t *> meaningful_videos;
      std::vector<const candidate_t *> meaningful_content;
      for (const auto &candidate : result.candidates) {
        if (candidate_area(candidate) < config.structural_competitor_min_area) {
          continue;
        }
        if (candidate.kind == roi_kind_e::video) {
          meaningful_videos.push_back(&candidate);
        } else if (candidate.kind == roi_kind_e::content) {
          meaningful_content.push_back(&candidate);
        }
      }

      auto block_videos = meaningful_videos.size() > 1;
      if (!block_videos && meaningful_videos.size() == 1 && !near_identity(*meaningful_videos.front())) {
        const auto video_area =
          candidate_area(*meaningful_videos.front());
        const auto matching_content = std::any_of(
          meaningful_content.begin(),
          meaningful_content.end(),
          [&](const candidate_t *content) {
            return candidate_intersection_area(
                     *meaningful_videos.front(),
                     *content
                   ) /
                     video_area >=
                   0.80f;
          }
        );
        const auto largest_content_area = std::accumulate(
          meaningful_content.begin(),
          meaningful_content.end(),
          0.0f,
          [](float largest, const candidate_t *content) {
            return std::max(largest, candidate_area(*content));
          }
        );
        block_videos =
          !matching_content ||
          video_area <
            largest_content_area *
              config.no_gutter_video_min_content_ratio;
      }
      const auto block_content =
        meaningful_videos.empty() && meaningful_content.size() > 1;
      if (!block_videos && !block_content) {
        return;
      }

      for (auto &candidate : result.candidates) {
        if (block_videos || (block_content && candidate.kind == roi_kind_e::content)) {
          candidate.primary_column_support = 0.0f;
          candidate.gutter_confidence = 0.0f;
          candidate.inside_primary_column = false;
        }
      }
    }

    bool canonical_candidate_less(
      const candidate_t &lhs,
      const candidate_t &rhs
    ) {
      return std::tie(
               lhs.kind,
               lhs.rect.left,
               lhs.rect.top,
               lhs.rect.right,
               lhs.rect.bottom
             ) <
             std::tie(
               rhs.kind,
               rhs.rect.left,
               rhs.rect.top,
               rhs.rect.right,
               rhs.rect.bottom
             );
    }

    void detect_scroll(
      feature_detection_result_t &result,
      feature_grid_view_t grid,
      const feature_detector_config_t &config,
      const feature_detector_context_t &context
    ) {
      const auto &exclusion = context.video_motion_exclusion;
      std::array<std::uint64_t, 256> shift_histogram {};
      std::uint64_t motion_weight = 0;
      std::size_t available_cells = 0;
      std::array<std::size_t, 3> horizontal_available {};
      std::array<std::size_t, 3> vertical_available {};

      for (std::uint16_t y = 0; y < grid.height; ++y) {
        for (std::uint16_t x = 0; x < grid.width; ++x) {
          if (exclusion && contains_cell(*exclusion, x, y, grid.width, grid.height)) {
            continue;
          }
          ++available_cells;
          const auto horizontal_band =
            std::min<std::size_t>(
              2,
              static_cast<std::size_t>(x) * 3 / grid.width
            );
          const auto vertical_band =
            std::min<std::size_t>(
              2,
              static_cast<std::size_t>(y) * 3 / grid.height
            );
          ++horizontal_available[horizontal_band];
          ++vertical_available[vertical_band];
          const auto &cell = grid.cells[cell_index(x, y, grid.width)];
          if (cell.vertical_shift_confidence_q8 < config.scroll_confidence_threshold_q8 || std::abs(static_cast<int>(cell.vertical_shift_rows)) < config.scroll_min_shift_rows || std::max(cell.temporal_occupancy_q8, cell.photographic_density_q8) < config.video_photo_threshold_q8) {
            continue;
          }

          const auto weight =
            static_cast<std::uint64_t>(cell.vertical_shift_confidence_q8);
          motion_weight += weight;
          shift_histogram[static_cast<int>(cell.vertical_shift_rows) + 128] += weight;
        }
      }

      if (available_cells == 0 || motion_weight == 0) {
        return;
      }

      int dominant_shift = 0;
      std::uint64_t dominant_window_weight = 0;
      for (int shift = -128; shift <= 127; ++shift) {
        if (std::abs(shift) < config.scroll_min_shift_rows) {
          continue;
        }
        std::uint64_t window_weight = 0;
        for (int neighbor = std::max(-128, shift - 1); neighbor <= std::min(127, shift + 1); ++neighbor) {
          window_weight += shift_histogram[neighbor + 128];
        }
        if (window_weight > dominant_window_weight || (window_weight == dominant_window_weight && std::pair {std::abs(shift), shift} < std::pair {
                                                                                                                                         std::abs(dominant_shift),
                                                                                                                                         dominant_shift,
                                                                                                                                       })) {
          dominant_shift = shift;
          dominant_window_weight = window_weight;
        }
      }
      if (dominant_window_weight == 0) {
        return;
      }

      std::uint64_t coherent_weight = 0;
      std::int64_t coherent_shift_weight = 0;
      std::size_t coherent_cells = 0;
      std::array<std::size_t, 3> horizontal_coherent {};
      std::array<std::size_t, 3> vertical_coherent {};
      for (std::uint16_t y = 0; y < grid.height; ++y) {
        for (std::uint16_t x = 0; x < grid.width; ++x) {
          if (exclusion && contains_cell(*exclusion, x, y, grid.width, grid.height)) {
            continue;
          }
          const auto &cell = grid.cells[cell_index(x, y, grid.width)];
          if (cell.vertical_shift_confidence_q8 < config.scroll_confidence_threshold_q8 || std::abs(static_cast<int>(cell.vertical_shift_rows) - dominant_shift) > 1 || std::max(cell.temporal_occupancy_q8, cell.photographic_density_q8) < config.video_photo_threshold_q8) {
            continue;
          }
          const auto weight =
            static_cast<std::uint64_t>(cell.vertical_shift_confidence_q8);
          coherent_weight += weight;
          coherent_shift_weight +=
            static_cast<std::int64_t>(cell.vertical_shift_rows) * weight;
          ++coherent_cells;
          ++horizontal_coherent[std::min<std::size_t>(
            2,
            static_cast<std::size_t>(x) * 3 / grid.width
          )];
          ++vertical_coherent[std::min<std::size_t>(
            2,
            static_cast<std::size_t>(y) * 3 / grid.height
          )];
        }
      }

      const auto support =
        static_cast<float>(coherent_cells) / available_cells;
      const auto agreement =
        static_cast<float>(coherent_weight) / motion_weight;
      std::size_t horizontal_coverage = 0;
      for (std::size_t band = 0; band < horizontal_coherent.size(); ++band) {
        if (horizontal_available[band] > 0 && static_cast<float>(horizontal_coherent[band]) / horizontal_available[band] >= config.scroll_min_band_support_fraction) {
          ++horizontal_coverage;
        }
      }
      std::size_t vertical_coverage = 0;
      for (std::size_t band = 0; band < vertical_coherent.size(); ++band) {
        if (vertical_available[band] > 0 && static_cast<float>(vertical_coherent[band]) / vertical_available[band] >= config.scroll_min_band_support_fraction) {
          ++vertical_coverage;
        }
      }

      result.vertical_scroll_rows =
        coherent_weight == 0 ?
          0.0f :
          static_cast<float>(coherent_shift_weight) / coherent_weight;
      result.scroll_confidence =
        std::clamp(
          std::min(
            support /
              std::max(config.scroll_min_support_fraction, 0.001f),
            1.0f
          ) *
            agreement,
          0.0f,
          1.0f
        );
      result.broad_page_scroll =
        support >= config.scroll_min_support_fraction &&
        agreement >= config.scroll_min_direction_agreement &&
        horizontal_coverage >= 2 &&
        vertical_coverage == 3;
    }

    void validate_config(const feature_detector_config_t &config) {
      if (config.max_cells == 0 || config.max_cells > 65536 || config.max_candidates == 0 || config.max_candidates > tracker_candidate_budget || config.max_components == 0 || config.max_components > 4096 || config.max_grid_dimension == 0 || config.max_grid_dimension > 4096 || config.min_gutter_width_cells == 0 || config.min_component_width_cells == 0 || config.min_component_height_cells == 0 || config.min_gutter_width_cells > 32 || config.video_bridge_x_cells > 32 || config.video_bridge_y_cells > 32 || config.content_bridge_x_cells > 32 || config.content_bridge_y_cells > 32 || config.content_halo_cells > 8 || config.min_component_width_cells > 32 || config.min_component_height_cells > 32) {
        throw std::invalid_argument("SBS ROI feature detector budgets must be bounded and non-zero");
      }

      const float fractions[] {
        config.gutter_row_fraction,
        config.content_trim_fraction,
        config.content_min_area,
        config.content_min_width,
        config.content_min_height,
        config.content_min_fill,
        config.content_axis_fill,
        config.content_min_axis_occupancy,
        config.structural_competitor_min_area,
        config.no_gutter_video_min_content_ratio,
        config.scroll_min_support_fraction,
        config.scroll_min_direction_agreement,
        config.scroll_min_band_support_fraction,
      };
      if (std::any_of(std::begin(fractions), std::end(fractions), [](float value) {
            return !valid_fraction(value);
          }) ||
          config.gutter_row_fraction <= 0.5f || config.content_trim_fraction >= 0.50f || config.content_min_area <= 0.0f || config.content_min_width <= 0.0f || config.content_min_height <= 0.0f || config.content_min_fill <= 0.0f || config.content_axis_fill <= 0.0f || config.content_min_axis_occupancy <= 0.0f || config.structural_competitor_min_area <= 0.0f || config.no_gutter_video_min_content_ratio <= 0.0f || config.scroll_min_support_fraction <= 0.0f || config.scroll_min_direction_agreement <= 0.5f || config.scroll_min_band_support_fraction <= 0.0f || !finite(config.primary_column_width_ratio) || config.primary_column_width_ratio <= 1.0f || !finite(config.content_trim_min_density_gain) || config.content_trim_min_density_gain <= 1.0f || !finite(config.content_max_aspect_ratio) || config.content_max_aspect_ratio <= 1.0f || config.scroll_min_shift_rows <= 0) {
        throw std::invalid_argument("SBS ROI feature detector thresholds are outside their safe ranges");
      }
    }
  }  // namespace

  feature_detector_t::feature_detector_t(feature_detector_config_t config):
      config_(std::move(config)) {
    validate_config(config_);
  }

  feature_detection_result_t feature_detector_t::detect(
    feature_grid_view_t grid,
    std::optional<normalized_point_t> recent_interaction,
    feature_detector_context_t context
  ) const {
    feature_detection_result_t result;
    const auto cell_count =
      static_cast<std::size_t>(grid.width) * grid.height;
    if (grid.width == 0 || grid.height == 0 || grid.width > config_.max_grid_dimension || grid.height > config_.max_grid_dimension || cell_count > config_.max_cells || grid.cells.size() != cell_count) {
      result.status = feature_detection_status_e::invalid_shape;
      return result;
    }
    if (recent_interaction && !valid_point(*recent_interaction)) {
      result.status = feature_detection_status_e::invalid_interaction;
      return result;
    }
    if (context.video_motion_exclusion && !valid_rect(*context.video_motion_exclusion)) {
      result.status = feature_detection_status_e::invalid_context;
      return result;
    }

    const auto gutters = find_structural_gutters(grid, config_);
    const auto barriers = barrier_columns(grid.width, gutters);
    const auto columns = find_columns(grid, gutters);
    bool primary_ambiguous = false;
    const auto primary_index =
      choose_primary_column(
        columns,
        config_.primary_column_width_ratio,
        primary_ambiguous
      );
    std::optional<cell_rect_t> primary_column;
    if (primary_index) {
      primary_column = columns[*primary_index].bounds;
      result.primary_column =
        normalize_rect(*primary_column, grid.width, grid.height);
    } else if (columns.size() == 1) {
      primary_column = columns.front().bounds;
      result.primary_column =
        normalize_rect(*primary_column, grid.width, grid.height);
    }
    result.primary_column_ambiguous = primary_ambiguous;

    result.gutters.reserve(gutters.size());
    for (const auto &gutter : gutters) {
      result.gutters.push_back({
        normalize_rect(gutter.bounds, grid.width, grid.height),
        gutter.confidence,
      });
    }
    result.columns.reserve(columns.size());
    for (std::size_t index = 0; index < columns.size(); ++index) {
      const auto cell_area =
        static_cast<std::uint64_t>(
          columns[index].bounds.right - columns[index].bounds.left
        ) *
        grid.height;
      result.columns.push_back({
        normalize_rect(columns[index].bounds, grid.width, grid.height),
        cell_area == 0 ?
          0.0f :
          static_cast<float>(columns[index].evidence) /
            (static_cast<float>(cell_area) * 255.0f),
        primary_index && *primary_index == index,
      });
    }

    std::vector<std::uint8_t> activity_seed_mask(cell_count, 0);
    std::vector<std::uint8_t> video_support_mask(cell_count, 0);
    std::vector<std::uint8_t> content_mask(cell_count, 0);
    for (std::size_t index = 0; index < cell_count; ++index) {
      if (barriers[index % grid.width] != 0) {
        continue;
      }
      const auto &cell = grid.cells[index];
      activity_seed_mask[index] =
        cell.temporal_occupancy_q8 >=
            config_.video_activity_threshold_q8 ?
          1 :
          0;
      video_support_mask[index] =
        activity_seed_mask[index] != 0 ||
            cell.photographic_density_q8 >=
              config_.video_photo_threshold_q8 ?
          1 :
          0;
      content_mask[index] =
        cell.photographic_density_q8 >=
            config_.content_photo_threshold_q8 ?
          1 :
          0;
    }

    const auto opened_video_support = open_video_support(
      video_support_mask,
      activity_seed_mask,
      grid.width,
      grid.height,
      barriers
    );
    const auto bridged_video_support = bridge_small_gaps(
      opened_video_support,
      grid.width,
      grid.height,
      scaled_cells(
        config_.video_bridge_x_cells,
        grid.width,
        nominal_feature_grid_width,
        true
      ),
      scaled_cells(
        config_.video_bridge_y_cells,
        grid.height,
        nominal_feature_grid_height,
        true
      ),
      barriers
    );
    const auto raw_content_components = collect_components(
      content_mask,
      grid.width,
      grid.height,
      config_.max_components
    );
    if (raw_content_components.overflow) {
      result.status = feature_detection_status_e::overflow;
      return result;
    }
    auto filtered_content_mask = content_mask;
    for (const auto &component : raw_content_components.components) {
      const auto width_fraction =
        static_cast<float>(
          component.bounds.right - component.bounds.left
        ) /
        grid.width;
      const auto height_fraction =
        static_cast<float>(
          component.bounds.bottom - component.bounds.top
        ) /
        grid.height;
      const auto aspect = std::max(
        width_fraction / std::max(height_fraction, 0.0001f),
        height_fraction / std::max(width_fraction, 0.0001f)
      );
      if (aspect <= config_.content_max_aspect_ratio) {
        continue;
      }
      for (auto y = component.bounds.top; y < component.bounds.bottom; ++y) {
        for (auto x = component.bounds.left; x < component.bounds.right; ++x) {
          const auto index = cell_index(x, y, grid.width);
          if (raw_content_components.labels[index] == component.label) {
            filtered_content_mask[index] = 0;
          }
        }
      }
    }
    const auto bridged_content = bridge_small_gaps(
      filtered_content_mask,
      grid.width,
      grid.height,
      scaled_cells(
        config_.content_bridge_x_cells,
        grid.width,
        nominal_feature_grid_width,
        true
      ),
      scaled_cells(
        config_.content_bridge_y_cells,
        grid.height,
        nominal_feature_grid_height,
        true
      ),
      barriers
    );
    const auto video_components = collect_components(
      bridged_video_support,
      grid.width,
      grid.height,
      config_.max_components
    );
    const auto content_components = collect_components(
      bridged_content,
      grid.width,
      grid.height,
      config_.max_components
    );
    if (video_components.overflow || content_components.overflow) {
      result.status = feature_detection_status_e::overflow;
      return result;
    }

    const auto min_width = scaled_cells(
      config_.min_component_width_cells,
      grid.width,
      nominal_feature_grid_width
    );
    const auto min_height = scaled_cells(
      config_.min_component_height_cells,
      grid.height,
      nominal_feature_grid_height
    );
    result.candidates.reserve(config_.max_candidates);
    for (const auto &component : video_components.components) {
      const auto width = component.bounds.right - component.bounds.left;
      const auto height = component.bounds.bottom - component.bounds.top;
      if (width < min_width || height < min_height || component_mask_cells(component, video_components, activity_seed_mask, grid.width) == 0) {
        continue;
      }
      result.candidates.push_back(
        make_candidate(
          roi_kind_e::video,
          component.bounds,
          grid,
          gutters,
          barriers,
          primary_column,
          primary_ambiguous,
          recent_interaction
        )
      );
      if (result.candidates.size() > config_.max_candidates) {
        result.status = feature_detection_status_e::overflow;
        result.candidates.clear();
        return result;
      }
    }

    const auto halo_x = scaled_cells(
      config_.content_halo_cells,
      grid.width,
      nominal_feature_grid_width,
      true
    );
    const auto halo_y = scaled_cells(
      config_.content_halo_cells,
      grid.height,
      nominal_feature_grid_height,
      true
    );
    for (const auto &component : content_components.components) {
      const auto width = component.bounds.right - component.bounds.left;
      const auto height = component.bounds.bottom - component.bounds.top;
      if (width < min_width || height < min_height || !content_component_eligible(component, content_components, filtered_content_mask, grid, config_)) {
        continue;
      }
      auto bounds = trim_content_bounds(
        component,
        content_components,
        grid,
        config_.content_trim_fraction,
        config_.content_trim_min_density_gain
      );
      bounds = add_content_halo(
        bounds,
        halo_x,
        halo_y,
        grid.width,
        grid.height,
        barriers
      );
      if (normalized_area(bounds, grid.width, grid.height) <= 0.0f) {
        continue;
      }
      result.candidates.push_back(
        make_candidate(
          roi_kind_e::content,
          bounds,
          grid,
          gutters,
          barriers,
          primary_column,
          primary_ambiguous,
          recent_interaction
        )
      );
      if (result.candidates.size() > config_.max_candidates) {
        result.status = feature_detection_status_e::overflow;
        result.candidates.clear();
        return result;
      }
    }

    std::sort(
      result.candidates.begin(),
      result.candidates.end(),
      canonical_candidate_less
    );
    fail_closed_without_structural_separation(result, config_);
    detect_scroll(result, grid, config_, context);
    return result;
  }
}  // namespace sbs_roi

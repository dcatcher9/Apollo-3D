// SPDX-License-Identifier: GPL-3.0-only
#include "scene_depth_source.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
  using namespace sunshine_scene_depth;

  void require(bool condition, const char *message) {
    if (!condition) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
  }
  // The FG source id as every site built it before source_key, pinned apart
  // from encode_source_id: logs and Dump 3D packages carry this value.
  constexpr std::uint64_t legacy_fg(std::uint32_t viewport) { return (std::uint64_t{1} << 63) | viewport; }
  constexpr std::uint32_t viewports[] = {0, 1, 2, 7, UINT32_MAX};
  // A key as a frame carries it: its provider, viewport and source_id.
  struct carried {
    provider_kind provider;
    source_key key;
  };
  // The test's inverse of encode_source_id: the key of a frame's provider,
  // viewport and source_id. A frame is never Generic's. No production code
  // decodes a source_id.
  constexpr source_key decode_source_id(provider_kind provider, std::uint32_t viewport, std::uint64_t id) {
    if (provider == provider_kind::streamline && id == legacy_fg(viewport)) return fg_depth_key(viewport);
    return {source_role::scene_depth, reporting_api(provider), viewport, id};
  }
}

int main() {
  static_assert(encode_source_id(fg_depth_key(2)) == legacy_fg(2), "FG depth's legacy source id changed");
  // Legacy encoding unchanged: FG depth (1 << 63) | viewport, Streamline
  // scene depth zero, NGX its feature generation.
  for (const auto viewport : viewports) {
    const auto key = fg_depth_key(viewport);
    require(key.role == source_role::fg_depth && key.api == source_api::streamline && key.viewport == viewport && !key.id,
      "An FG key names more than its role, Streamline and its viewport");
    require(encode_source_id(key) == legacy_fg(viewport), "FG depth's legacy source id changed");
    require(!encode_source_id({source_role::scene_depth, source_api::streamline, viewport}),
      "Streamline scene depth's legacy source id is no longer zero");
    for (const std::uint64_t generation : {std::uint64_t{1}, std::uint64_t{41}})
      require(encode_source_id({source_role::scene_depth, source_api::ngx, viewport, generation}) == generation,
        "NGX's legacy source id is no longer its feature generation");
  }
  // Every key a frame carries today, per viewport: FG and scene depth from
  // Streamline, and NGX features, one of whose generations equals that
  // viewport's FG id.
  std::vector<carried> keys;
  for (const auto viewport : viewports) {
    keys.push_back({provider_kind::streamline, fg_depth_key(viewport)});
    keys.push_back({provider_kind::streamline, {source_role::scene_depth, source_api::streamline, viewport}});
    for (const std::uint64_t generation : {std::uint64_t{1}, std::uint64_t{41}, legacy_fg(viewport)})
      keys.push_back({provider_kind::ngx, {source_role::scene_depth, source_api::ngx, viewport, generation}});
  }
  // Round trip: a frame's provider, viewport and source_id name its key.
  for (const auto &value : keys)
    require(reporting_api(value.provider) == value.key.api &&
        decode_source_id(value.provider, value.key.viewport, encode_source_id(value.key)) == value.key,
      "A frame's provider, viewport and source_id no longer name its key");
  // Unique per role, API and viewport: two keys never share a frame's
  // provider, viewport and source_id.
  for (std::size_t i = 0; i != keys.size(); ++i)
    for (std::size_t j = i + 1; j != keys.size(); ++j) {
      const auto &a = keys[i], &b = keys[j];
      require(a.key != b.key, "Two fixture keys are equal");
      require(a.provider != b.provider || a.key.viewport != b.key.viewport ||
          encode_source_id(a.key) != encode_source_id(b.key),
        "Two sources share one provider, viewport and source_id");
    }
  // Decoding is exact: any value re-encodes to itself, and only the frame's
  // own viewport's FG id from Streamline is FG depth.
  for (const auto provider : {provider_kind::streamline, provider_kind::ngx})
    for (const auto viewport : viewports) {
      const std::uint64_t ids[] = {0, 1, legacy_fg(0), legacy_fg(viewport), legacy_fg(viewport) | (std::uint64_t{1} << 40), UINT64_MAX};
      for (const auto id : ids) {
        const auto key = decode_source_id(provider, viewport, id);
        require(encode_source_id(key) == id && key.viewport == viewport && key.api == reporting_api(provider),
          "A decoded key does not re-encode to its source_id");
        require((key.role == source_role::fg_depth) == (provider == provider_kind::streamline && id == legacy_fg(viewport)),
          "A source_id decoded as FG depth outside its own viewport or Streamline");
      }
    }
  std::puts("PASS source keys: legacy encoding, round trip, uniqueness per role, API and viewport, exact decoding");
}

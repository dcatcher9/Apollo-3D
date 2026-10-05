// SPDX-License-Identifier: GPL-3.0-only
// GPU contract of the UI selection predicate (UI framework S1, S2a, S2b, fix 1, fix 2, S3): runs the
// real SunshineUIDetectionReduceCS of game3d_native.hlsl on synthetic
// per-tile statistics, detection constants and T1 hold stores, and compares
// every decision word, the written hold store and every counter add with
// ui_selection::decide (game3d_ui_selection.h). The tiles pass's alpha,
// layer, HUD-less difference, one-way judgment and pre-UI pixel counts are
// checked against a CPU count of V1, V2, A2 and the layer's pre-UI comparison
// (H1 d) on edge values, and the scene compare and evidence passes' H2
// stillness counts (statistics rows 144-152, decision texel 12, the previous
// cell means at u5) against a CPU oracle. Also checks the predicate's
// intended behaviour (selection, the T1 grace, F1 reasons and refused
// candidates, the H1 override of a hidden scene and its informative claims,
// the H2 override of a still screen without a UI source), pair
// comparability (V2) and the acceptance key (A1). S3: the identity verdicts
// of texel 12 .z/.w and the identity counter words on synthetic stamps (t9)
// and proposals (b2 words 6-9), and the HUD-less gate of sources 5 and 6.
// Fix 3 and fix 4 (selection revision 6) were removed by user decision;
// revision 7 decides as revision 5, and revision 8 adds the empty change set.
// Uses a hardware D3D11 device, else WARP.
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include "game3d_ui_selection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
  using Microsoft::WRL::ComPtr;
  namespace detection = sunshine_game3d::ui_detection;
  namespace selection = sunshine_game3d::ui_selection;
  namespace candidate = detection::candidate;
  namespace word = detection::decision_word;
  using sunshine_game3d::ui_counter_word::count;
  using sunshine_game3d::ui_counter_word::with_identity;
  namespace identity = detection::identity;

  void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
  }

  void checked(HRESULT result, const char *what) {
    if (FAILED(result)) {
      char text[96];
      std::snprintf(text, sizeof(text), "%s failed: 0x%08lx", what, static_cast<unsigned long>(result));
      throw std::runtime_error(text);
    }
  }

  using texel = std::array<std::uint32_t, 4>;
  // Per-tile statistics rows as SunshineUIDetectionTilesCS writes them, one
  // texel per 16x16 tile (lane = row * 16 + column).
  struct tiles {
    std::array<texel, 256> coverage{}, invalid{}, difference{}, lit{}, layer{}, strong{}, contradicted{};
    // H1 (d), rows 128-143: the layer against the presented frame.
    std::array<texel, 256> pre_ui{};
  };

  // What the reduce sums from the tiles, exactly as it does.
  selection::counts sums(const tiles &t) {
    selection::counts c;
    std::array<std::uint32_t, 4> coverage{}, invalid{}, difference{};
    std::array<std::uint32_t, 4> lit{}, layer{}, pre_ui{};
    for (std::size_t i = 0; i != 256; ++i) {
      for (std::size_t k = 0; k != 4; ++k) {
        coverage[k] += t.coverage[i][k];
        invalid[k] += t.invalid[i][k];
        difference[k] += t.difference[i][k];
        lit[k] += t.lit[i][k];
        layer[k] += t.layer[i][k];
        pre_ui[k] += t.pre_ui[i][k];
      }
      for (std::size_t k = 0; k != 3; ++k) {
        c.strong[k] += t.strong[i][k];
        c.contradicted[k] += t.contradicted[i][k];
      }
      const auto &d = t.difference[i];
      c.matching_tiles += d[3] && d[2] * 100u >= d[3] * 99u ? 1u : 0u;
    }
    c.pixels = difference[3];
    c.changed = difference[0];
    c.nonfinite = difference[1];
    c.unchanged = difference[2];
    c.lit = lit[0];
    c.opaque_ui_alpha = lit[1];
    c.opaque_ui_color = lit[2];
    c.covered = {coverage[0], coverage[1], layer[0], coverage[2], coverage[3]};
    c.invalid = {invalid[0], invalid[1], layer[1], invalid[2], invalid[3]};
    c.opaque_layer = layer[2];
    c.opaque_backbuffer = lit[3];
    c.opaque_current = layer[3];
    c.pre_ui_match = pre_ui[0];
    c.pre_ui_lit = pre_ui[1];
    c.presented_lit = pre_ui[2];
    c.presented_lit_differs = pre_ui[3];
    return c;
  }

  struct case_t {
    std::string name;
    tiles statistics;
    std::uint32_t offered{}, accepted{}, flags{};
    // The hold store before the reduce; none keeps what the previous case's
    // reduce wrote (a chained real frame).
    std::optional<selection::hold_state> previous = selection::hold_state{};
    // b2 word 4: the renderer pushes the pre-UI threshold on sample frames
    // only; zero skips the pre-UI sums and writes texel 11 as zero.
    float pre_ui_threshold = 2.f / 255.f;
    // b2 word 5: H2's still-screen flag (still::flatten while the CPU's run
    // is active and enabled).
    std::uint32_t still{};
    // S3: the stamps at t9 and b2 words 6-9.
    selection::identity_input identity{};
  };

  // An even split of every pixel over the tiles.
  std::array<std::uint32_t, 256> split(std::uint32_t total) {
    std::array<std::uint32_t, 256> parts{};
    for (auto &part : parts) part = total / 256u;
    parts[0] += total % 256u;
    return parts;
  }
  // Puts per-kind totals into one tile row component, spread unevenly.
  void spread(std::array<texel, 256> &row, std::size_t component, std::uint32_t total, std::mt19937 &random) {
    std::uint32_t left = total;
    for (std::size_t i = 255; i != 0 && left; --i) {
      const std::uint32_t part = std::uniform_int_distribution<std::uint32_t>(0, left / 64u + 1u)(random);
      const std::uint32_t taken = part > left ? left : part;
      row[i][component] += taken;
      left -= taken;
    }
    row[0][component] += left;
  }

  // Difference rows: matching tiles fully unchanged, the others with the
  // given per-tile changed and unchanged shares (in 1/1000 of the tile).
  void difference_rows(tiles &t, std::uint32_t pixels, std::uint32_t matching, std::uint32_t changed_permille,
      std::uint32_t unchanged_permille) {
    const auto w = split(pixels);
    for (std::uint32_t i = 0; i != 256; ++i) {
      auto &d = t.difference[i];
      d[3] = w[i];
      if (i < matching) {
        d = {0, 0, w[i], w[i]};
      } else {
        d[0] = std::uint32_t(std::uint64_t(w[i]) * changed_permille / 1000u);
        d[2] = std::uint32_t(std::uint64_t(w[i]) * unchanged_permille / 1000u);
        d[1] = 0;
      }
    }
  }

  struct gpu_t {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Texture2D> statistics, decision, counters, hold, decision_staging, counters_staging, hold_staging;
    ComPtr<ID3D11ShaderResourceView> statistics_view;
    ComPtr<ID3D11UnorderedAccessView> decision_view, counters_view, hold_view;
    ComPtr<ID3D11Buffer> constants;
    // S3: the stamp buffer at t9 (identity::stamp_entries uint4 entries).
    ComPtr<ID3D11Buffer> stamps;
    ComPtr<ID3D11ShaderResourceView> stamps_view;
    std::string adapter;
    // The hold store as the last reduce wrote it.
    selection::hold_state hold_written{};
  };

  gpu_t create_gpu() {
    gpu_t gpu;
    const D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
    HRESULT result = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &level, 1, D3D11_SDK_VERSION, &gpu.device,
      nullptr, &gpu.context);
    gpu.adapter = "hardware";
    if (FAILED(result)) {
      result = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, &level, 1, D3D11_SDK_VERSION, &gpu.device, nullptr,
        &gpu.context);
      gpu.adapter = "WARP";
    }
    checked(result, "D3D11CreateDevice");
    D3D11_TEXTURE2D_DESC desc{};
    desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
    desc.Width = detection::statistics_columns(detection::tile_parts);
    desc.Height = detection::statistics_rows(0);
    desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.statistics), "statistics");
    checked(gpu.device->CreateShaderResourceView(gpu.statistics.Get(), nullptr, &gpu.statistics_view), "statistics view");
    desc.Width = detection::still_decision_texels;
    desc.Height = 1;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.decision), "decision");
    checked(gpu.device->CreateUnorderedAccessView(gpu.decision.Get(), nullptr, &gpu.decision_view), "decision view");
    desc.BindFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.decision_staging), "decision staging");
    desc.Width = UINT(with_identity);
    desc.Format = DXGI_FORMAT_R32_UINT;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.CPUAccessFlags = 0;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.counters), "counters");
    checked(gpu.device->CreateUnorderedAccessView(gpu.counters.Get(), nullptr, &gpu.counters_view), "counters view");
    desc.BindFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.counters_staging), "counters staging");
    // The T1 hold store, R32_UINT like the renderer's.
    desc.Width = detection::hold::store_texels;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.CPUAccessFlags = 0;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.hold), "hold store");
    checked(gpu.device->CreateUnorderedAccessView(gpu.hold.Get(), nullptr, &gpu.hold_view), "hold store view");
    desc.BindFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.hold_staging), "hold store staging");
    D3D11_BUFFER_DESC buffer{};
    buffer.ByteWidth = 48; // b2: ten words, padded to 16 bytes.
    buffer.Usage = D3D11_USAGE_DEFAULT;
    buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    checked(gpu.device->CreateBuffer(&buffer, nullptr, &gpu.constants), "constants");
    buffer.ByteWidth = identity::stamp_entries * 16u;
    buffer.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    checked(gpu.device->CreateBuffer(&buffer, nullptr, &gpu.stamps), "stamps");
    D3D11_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R32G32B32A32_UINT;
    view.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    view.Buffer.NumElements = identity::stamp_entries;
    checked(gpu.device->CreateShaderResourceView(gpu.stamps.Get(), &view, &gpu.stamps_view), "stamps view");
    return gpu;
  }

  ComPtr<ID3D11ComputeShader> compile_pass(gpu_t &gpu, const std::string &source, unsigned color,
      const char *entry = "SunshineUIDetectionReduceCS") {
    const auto c = std::to_string(color);
    const D3D_SHADER_MACRO defines[]{{"BUFFER_WIDTH", "256"}, {"BUFFER_HEIGHT", "144"}, {"BUFFER_COLOR_SPACE", c.c_str()},
      {nullptr, nullptr}};
    ComPtr<ID3DBlob> code, errors;
    const HRESULT result = D3DCompile(source.data(), source.size(), "game3d_native.hlsl", defines, nullptr, entry, "cs_5_0",
      D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
    if (FAILED(result))
      throw std::runtime_error(std::string(entry) + ": " +
        (errors ? static_cast<const char *>(errors->GetBufferPointer()) : "compile failed"));
    ComPtr<ID3D11ComputeShader> shader;
    checked(gpu.device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader), "shader");
    return shader;
  }

  // Every texel of a texture, row by row.
  template<class T> std::vector<T> read(gpu_t &gpu, ID3D11Texture2D *source, ID3D11Texture2D *staging, std::size_t values) {
    D3D11_TEXTURE2D_DESC desc{};
    staging->GetDesc(&desc);
    gpu.context->CopyResource(staging, source);
    D3D11_MAPPED_SUBRESOURCE mapped{};
    checked(gpu.context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped), "map");
    std::vector<T> result(values);
    const std::size_t per_row = values / desc.Height;
    for (std::size_t row = 0; row != desc.Height; ++row)
      std::memcpy(result.data() + row * per_row, static_cast<const char *>(mapped.pData) + row * mapped.RowPitch, per_row * sizeof(T));
    gpu.context->Unmap(staging, 0);
    return result;
  }

  // b2 as the renderer pushes it (ten words, S3's proposals and identity
  // bits in words 6-9), padded to the buffer's 48 bytes.
  struct detection_constants {
    std::uint32_t offered;
    float threshold;
    std::uint32_t accepted, flags;
    float pre_ui_threshold;
    std::uint32_t still;
    std::uint32_t padding[6];
  };
  static_assert(sizeof(detection_constants) == 48);
  // S3: b2 words 6-9 of an identity input, in the padding.
  detection_constants with_identity_words(detection_constants c, const selection::identity_input &in) {
    c.padding[0] = in.expected_layer_present;
    c.padding[1] = in.expected_layer_token;
    c.padding[2] = in.expected_hudless_present;
    c.padding[3] = in.bits;
    return c;
  }
  // The stamp entries an identity input reads: the layer's {C_P, C_T} and
  // the HUD-less image's C_P; every other entry a sentinel no pair reads.
  void upload_stamps(gpu_t &gpu, const selection::identity_input &in) {
    std::array<texel, identity::stamp_entries> entries{};
    for (auto &entry : entries) entry = {0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
    entries[identity::stamp_layer] = {in.layer_present, in.layer_token, 0u, 0u};
    entries[identity::stamp_hudless] = {in.hudless_present, 0x5eedu, 0u, 0u};
    gpu.context->UpdateSubresource(gpu.stamps.Get(), 0, nullptr, entries.data(), 0, 0);
  }

  // Runs one case and compares it; returns the decision for the semantic checks.
  selection::decision run(gpu_t &gpu, ID3D11ComputeShader *reduce, const case_t &test, const char *space) {
    // Each tile's counts split over its tile parts (columns 16 apart) as the
    // tiles pass writes them: the reduce must add them back exactly.
    constexpr std::size_t columns = detection::statistics_columns(detection::tile_parts);
    std::vector<texel> rows(columns * detection::statistics_rows(0));
    for (std::size_t lane = 0; lane != 256; ++lane) {
      const std::size_t column = lane % 16, row = lane / 16;
      const auto put = [&](std::size_t first_row, const texel &value) {
        for (std::size_t part = 0; part != detection::tile_parts; ++part)
          for (std::size_t k = 0; k != 4; ++k) {
            const std::uint32_t share = value[k] / detection::tile_parts;
            rows[(first_row + row) * columns + part * 16 + column][k] =
              part ? share : value[k] - share * (detection::tile_parts - 1);
          }
      };
      put(0, test.statistics.coverage[lane]);
      put(16, test.statistics.invalid[lane]);
      put(32, test.statistics.difference[lane]);
      put(48, test.statistics.lit[lane]);
      put(detection::layer_statistics_row, test.statistics.layer[lane]);
      put(detection::judgment_statistics_row, test.statistics.strong[lane]);
      put(detection::judgment_statistics_row + 16, test.statistics.contradicted[lane]);
      put(detection::pre_ui_statistics_row, test.statistics.pre_ui[lane]);
    }
    gpu.context->UpdateSubresource(gpu.statistics.Get(), 0, nullptr, rows.data(), UINT(columns * sizeof(texel)), 0);
    // The reduce reads b2 word 4 only as zero or not (texel 11 sums or zero).
    const auto constants = with_identity_words({test.offered, 2.f / 255.f, test.accepted, test.flags, test.pre_ui_threshold,
      test.still, {}}, test.identity);
    gpu.context->UpdateSubresource(gpu.constants.Get(), 0, nullptr, &constants, 0, 0);
    upload_stamps(gpu, test.identity);
    const std::array<std::uint32_t, 4> sentinel{0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
    gpu.context->ClearUnorderedAccessViewUint(gpu.decision_view.Get(), sentinel.data());
    const std::array<std::uint32_t, 4> zero{};
    gpu.context->ClearUnorderedAccessViewUint(gpu.counters_view.Get(), zero.data());
    // The hold store this real frame finds: the case's, or the previous one's.
    const auto previous = test.previous ? *test.previous : gpu.hold_written;
    if (test.previous) {
      const std::array<std::uint32_t, detection::hold::store_texels> store{previous.state, previous.source, previous.covered};
      gpu.context->UpdateSubresource(gpu.hold.Get(), 0, nullptr, store.data(), UINT(sizeof(store)), 0);
    }
    gpu.context->CSSetShader(reduce, nullptr, 0);
    ID3D11ShaderResourceView *views[11]{};
    views[9] = gpu.stamps_view.Get();
    views[10] = gpu.statistics_view.Get();
    gpu.context->CSSetShaderResources(0, 11, views);
    ID3D11Buffer *buffers[3]{nullptr, nullptr, gpu.constants.Get()};
    gpu.context->CSSetConstantBuffers(0, 3, buffers);
    // u5 the hold store, u6 the decision texels, u7 the counters.
    ID3D11UnorderedAccessView *uavs[3]{gpu.hold_view.Get(), gpu.decision_view.Get(), gpu.counters_view.Get()};
    gpu.context->CSSetUnorderedAccessViews(5, 3, uavs, nullptr);
    gpu.context->Dispatch(1, 1, 1);
    ID3D11UnorderedAccessView *none[3]{};
    gpu.context->CSSetUnorderedAccessViews(5, 3, none, nullptr);
    const auto words = read<std::uint32_t>(gpu, gpu.decision.Get(), gpu.decision_staging.Get(),
      4 * detection::still_decision_texels);
    const auto counters = read<std::uint32_t>(gpu, gpu.counters.Get(), gpu.counters_staging.Get(), with_identity);
    const auto store = read<std::uint32_t>(gpu, gpu.hold.Get(), gpu.hold_staging.Get(), detection::hold::store_texels);
    gpu.hold_written = {store[detection::hold::word::state], store[detection::hold::word::source],
      store[detection::hold::word::covered]};

    const auto c = sums(test.statistics);
    const auto d = selection::decide(c, test.offered, test.accepted, test.flags, previous, test.still, test.identity);
    // Texel 12 (H2's stillness counts) reads zero: the reduce clears it for
    // the evidence passes; .z/.w are S3's identity words.
    std::array<std::uint32_t, 4 * detection::still_decision_texels> want{};
    want[word::id_verdicts] = selection::identity_verdict_word(d.identity);
    want[word::id_deltas] = selection::identity_delta_word(d.identity);
    want[word::source] = d.source;
    want[word::covered] = d.covered;
    want[word::pixels] = c.pixels;
    want[word::matching_tiles] = c.matching_tiles;
    want[word::candidates] = test.offered;
    want[word::hudless_changed] = c.changed;
    want[word::hudless_unchanged] = c.unchanged;
    want[word::hudless_invalid] = c.nonfinite;
    want[word::alpha_covered] = c.covered[0];
    want[word::alpha_covered + 1] = c.covered[1];
    want[word::alpha_covered + 2] = c.covered[3];
    want[word::alpha_covered + 3] = c.covered[4];
    want[word::alpha_invalid] = c.invalid[0];
    want[word::alpha_invalid + 1] = c.invalid[1];
    want[word::alpha_invalid + 2] = c.invalid[3];
    want[word::alpha_invalid + 3] = c.invalid[4];
    want[word::hudless_lit] = c.lit;
    want[word::accepted] = test.accepted;
    want[word::alpha_opaque] = c.opaque_ui_alpha;
    want[word::alpha_opaque + 1] = c.opaque_ui_color;
    want[word::layer_covered] = c.covered[2];
    want[word::layer_invalid] = c.invalid[2];
    want[word::layer_opaque] = c.opaque_layer;
    want[word::valid_bits] = d.valid_bits;
    for (std::size_t i = 0; i != selection::judged_kinds.size(); ++i) {
      want[word::strong + i] = c.strong[i];
      want[word::contradicted + i] = c.contradicted[i];
    }
    want[word::refused] = d.refused;
    want[word::frame_reason] = selection::frame_reason_word(d);
    // Texel 10 (H1); the reduce zeroes the scene evidence texels 5 and 6.
    want[word::opaque_backbuffer] = c.opaque_backbuffer;
    want[word::opaque_current] = c.opaque_current;
    want[word::claims] = d.claims;
    want[word::h1] = selection::h1_word(d);
    // Texel 11 (H1 d): the layer's pre-UI pixel counts, zero without a layer
    // or a pre-UI threshold (a frame that is not a detection sample).
    const bool layer = (test.offered & candidate::layer) != 0u && test.pre_ui_threshold > 0.f;
    want[word::pre_ui_match] = layer ? c.pre_ui_match : 0u;
    want[word::pre_ui_image_lit] = layer ? c.pre_ui_lit : 0u;
    want[word::presented_lit] = layer ? c.presented_lit : 0u;
    want[word::presented_lit_differs] = layer ? c.presented_lit_differs : 0u;
    // decide() reads its counts back from the words it is compared with.
    const auto back = selection::counts_from_words(want.data(), want.size());
    require(back.covered == c.covered && back.invalid == c.invalid && back.pixels == c.pixels &&
        back.matching_tiles == c.matching_tiles && back.opaque_layer == c.opaque_layer && back.strong == c.strong &&
        back.contradicted == c.contradicted && back.opaque_ui_alpha == c.opaque_ui_alpha &&
        back.opaque_ui_color == c.opaque_ui_color && back.opaque_backbuffer == c.opaque_backbuffer &&
        back.opaque_current == c.opaque_current && back.pre_ui_match == want[word::pre_ui_match] &&
        back.pre_ui_lit == want[word::pre_ui_image_lit] && back.presented_lit == want[word::presented_lit] &&
        back.presented_lit_differs == want[word::presented_lit_differs] &&
        selection::identity_of_words(back.identity_verdicts, back.identity_deltas) == d.identity,
      test.name + ": counts_from_words does not invert the decision words");
    for (std::size_t i = 0; i != want.size(); ++i)
      if (words[i] != want[i]) {
        char text[240];
        std::snprintf(text, sizeof(text), " (%s): word %zu is %u on the GPU, decide() gives %u (offered 0x%x accepted 0x%x flags 0x%x "
          "still 0x%x hold %u/%u/%u)", space, i, words[i], want[i], test.offered, test.accepted, test.flags, test.still,
          previous.state, previous.source, previous.covered);
        throw std::runtime_error(test.name + text);
      }
    if (gpu.hold_written.state != d.next.state || gpu.hold_written.source != d.next.source ||
        gpu.hold_written.covered != d.next.covered) {
      char text[200];
      std::snprintf(text, sizeof(text), " (%s): the GPU wrote hold %u/%u/%u, decide() gives %u/%u/%u", space,
        gpu.hold_written.state, gpu.hold_written.source, gpu.hold_written.covered, d.next.state, d.next.source, d.next.covered);
      throw std::runtime_error(test.name + text);
    }
    const auto core = selection::counter_adds(d, test.flags);
    const auto identity_adds = selection::identity_counter_adds(d.identity);
    std::array<std::uint32_t, with_identity> adds{};
    std::copy(core.begin(), core.end(), adds.begin());
    std::copy(identity_adds.begin(), identity_adds.end(), adds.begin() + count);
    for (std::size_t i = 0; i != with_identity; ++i)
      if (counters[i] != adds[i]) {
        char text[200];
        std::snprintf(text, sizeof(text), " (%s): counter word %zu added %u on the GPU, decide() gives %u (source %u)", space, i,
          counters[i], adds[i], d.source);
        throw std::runtime_error(test.name + text);
      }
    return d;
  }

  // P = 256 x 144, the compiled detection size; the reduce never reads it.
  constexpr std::uint32_t small_pixels = 36864u;

  case_t alpha_case(std::string name, std::uint32_t pixels, std::uint32_t offered, std::uint32_t accepted, std::uint32_t flags = 0) {
    case_t test{std::move(name), {}, offered, accepted, flags};
    difference_rows(test.statistics, pixels, 0, 0, 0);
    return test;
  }
  // Sets one alpha kind's totals (in tile 0) by alpha index.
  void alpha(case_t &test, selection::kind k, std::uint32_t covered, std::uint32_t invalid, std::uint32_t opaque = 0) {
    auto &t = test.statistics;
    switch (k) {
      case selection::kind::ui_alpha: t.coverage[0][0] = covered; t.invalid[0][0] = invalid; t.lit[0][1] = opaque; break;
      case selection::kind::ui_color: t.coverage[0][1] = covered; t.invalid[0][1] = invalid; t.lit[0][2] = opaque; break;
      case selection::kind::backbuffer: t.coverage[0][2] = covered; t.invalid[0][2] = invalid; t.lit[0][3] = opaque; break;
      case selection::kind::current: t.coverage[0][3] = covered; t.invalid[0][3] = invalid; t.layer[0][3] = opaque; break;
      case selection::kind::ui_layer:
        t.layer[0][0] = covered;
        t.layer[0][1] = invalid;
        t.layer[0][2] = opaque;
        break;
      default: break;
    }
  }
  // Sets one judged kind's one-way counts (A2, in tile 0).
  void judgment(case_t &test, selection::kind k, std::uint32_t strong, std::uint32_t contradicted) {
    for (std::size_t i = 0; i != selection::judged_kinds.size(); ++i)
      if (selection::judged_kinds[i] == k) {
        test.statistics.strong[0][i] = strong;
        test.statistics.contradicted[0][i] = contradicted;
      }
  }

  // S3: the identity verdicts and the HUD-less gate on an offered layer
  // (its pair verdict decides nothing: no gate reads it since the pre-UI
  // change set was removed) and an
  // exact-or-not HUD-less pair (E33-like, a shapeless Backbuffer offered but
  // not accepted).
  // Without a gate every verdict decides nothing; with gate_hudless the
  // HUD-less change set is valid only with its pair exact (a token batch, or
  // the stamp's C_P + 1 equal to the proposal).
  std::vector<case_t> identity_cases() {
    std::vector<case_t> cases;
    const std::uint32_t p = small_pixels;
    using k = selection::kind;
    struct stamp_case {
      const char *name;
      selection::identity_input identity;
    };
    const stamp_case layer_stamps[]{
      {"exact present", {41u, 7u, 0u, 41u, 0u, 0u}},
      {"late copy (mismatch)", {42u, 7u, 0u, 41u, 0u, 0u}},
      {"unstamped", {0u, 0u, 0u, 41u, 0u, 0u}},
      {"unproposed", {41u, 7u, 0u, 0u, 0u, 0u}},
      {"exact token", {41u, 9u, 0u, 0u, 9u, 0u}},
      {"token mismatch", {41u, 8u, 0u, 0u, 9u, 0u}},
      {"far mismatch", {90000u, 7u, 0u, 41u, 0u, 0u}},
    };
    for (const auto &stamp : layer_stamps)
      for (const std::uint32_t gate : {0u, unsigned(identity::gate_hudless)}) {
        case_t test{std::string("S3 layer pair ") + stamp.name + ", gate " + std::to_string(gate), {},
          candidate::layer | candidate::current, candidate::layer | candidate::current, 0u};
        difference_rows(test.statistics, p, 128, 100, 900);
        test.statistics.lit[0][0] = p / 4u;
        alpha(test, k::ui_layer, 2000, 0, 100);
        alpha(test, k::current, p, 0, p);
        test.identity = stamp.identity;
        test.identity.bits = gate;
        cases.push_back(test);
      }
    const stamp_case hudless_stamps[]{
      {"token batch", {0u, 0u, 0u, 0u, 0u, 0u, identity::token_batch}},
      {"exact present", {0u, 0u, 40u, 0u, 0u, 41u}},
      {"generated Present (mismatch)", {0u, 0u, 41u, 0u, 0u, 41u}},
      {"unstamped", {0u, 0u, 0u, 0u, 0u, 41u}},
      {"unproposed", {0u, 0u, 40u, 0u, 0u, 0u}},
    };
    for (const auto &stamp : hudless_stamps)
      for (const std::uint32_t gate : {0u, unsigned(identity::gate_hudless)})
        for (const std::uint32_t exact : {0u, unsigned(candidate::exact)}) {
          case_t test{std::string("S3 HUD-less pair ") + stamp.name + ", gate " + std::to_string(gate) + ", exact " +
            std::to_string(exact), {}, candidate::backbuffer | candidate::hudless | exact | candidate::layer,
            candidate::hudless, 0u};
          difference_rows(test.statistics, p, 128, 100, 900);
          test.statistics.lit[0][0] = p;
          alpha(test, k::backbuffer, p, 0, p);
          alpha(test, k::ui_layer, 0, 30000);
          test.identity = stamp.identity;
          test.identity.bits |= gate;
          cases.push_back(test);
        }
    // A full change set (menus): a gated mismatched pair is not valid, so T1
    // reuses the previous own decision once.
    for (const std::uint32_t gate : {0u, unsigned(identity::gate_hudless)}) {
      case_t test{"S3 full HUD-less set, mismatch, gate " + std::to_string(gate), {}, candidate::hudless | candidate::exact,
        candidate::hudless, 0u};
      difference_rows(test.statistics, p, 0, 1000, 0);
      test.statistics.lit[0][0] = p;
      test.identity = {0u, 0u, 41u, 0u, 0u, 41u, gate};
      test.previous = selection::hold_state{detection::hold::own, 5u, 1234u};
      cases.push_back(test);
    }
    return cases;
  }

  std::vector<case_t> crafted() {
    std::vector<case_t> cases;
    const std::uint32_t p = small_pixels;
    using k = selection::kind;
    const auto hidden = detection::per_frame_scene_hidden, pre_ui = detection::per_frame_pre_ui_visible;
    const auto refuted = [](std::uint32_t bits) { return bits << detection::per_frame_refuted_shift; };
    // V1: 1% invalid is valid, one more pixel is not (p / 100 = 368.64).
    for (const std::uint32_t invalid : {368u, 369u}) {
      auto test = alpha_case("V1 bound, " + std::to_string(invalid) + " invalid", p, candidate::backbuffer, candidate::backbuffer);
      alpha(test, k::backbuffer, 500, invalid);
      cases.push_back(test);
      auto layer = alpha_case("V1 layer bound, " + std::to_string(invalid) + " invalid", p, candidate::layer, candidate::layer);
      alpha(layer, k::ui_layer, 500, invalid);
      cases.push_back(layer);
    }
    // The 90% selective bound (0.9 p = 33177.6), accepted and not.
    for (const std::uint32_t covered : {33177u, 33178u})
      for (const std::uint32_t accepted : {0u, unsigned(candidate::current)}) {
        auto test = alpha_case("90% bound, " + std::to_string(covered) + " covered", p, candidate::current, accepted);
        alpha(test, k::current, covered, 0);
        cases.push_back(test);
      }
    // A partial change set: 128 matching tiles (fully unchanged) and 128 at
    // 10% changed / 90% unchanged; 127 matching tiles fail.
    for (const std::uint32_t matching : {127u, 128u}) {
      case_t test{"128 matching tiles, " + std::to_string(matching), {}, candidate::hudless | candidate::exact, candidate::hudless};
      difference_rows(test.statistics, p, matching, 100, 900);
      cases.push_back(test);
    }
    // 75% unchanged: 128 clean tiles and 128 at 50% unchanged (5% changed)
    // reach exactly 75%; one fewer unchanged pixel fails.
    for (const std::uint32_t less : {0u, 1u}) {
      case_t test{"75% unchanged, minus " + std::to_string(less), {}, candidate::hudless, candidate::hudless};
      difference_rows(test.statistics, p, 128, 50, 500);
      test.statistics.difference[200][2] -= less;
      cases.push_back(test);
    }
    // 98% changed full sets from an exact pair, lit; accepted and not, with
    // and without the H1 holds and the HUD-less refuted, and inexact.
    for (const std::uint32_t permille : {979u, 980u, 1000u})
      for (const std::uint32_t accepted : {0u, unsigned(candidate::hudless)})
        for (const std::uint32_t flags : {0u, hidden, hidden | pre_ui, hidden | refuted(candidate::hudless)})
          for (const std::uint32_t exact : {0u, unsigned(candidate::exact)}) {
            case_t test{"full change set " + std::to_string(permille), {}, candidate::hudless | exact, accepted, flags};
            difference_rows(test.statistics, p, 0, permille, 0);
            test.statistics.lit[0][0] = p / 2u;
            cases.push_back(test);
          }
    // An empty change set (revision 8): a lit pair whose tiles changed no
    // pixel, at least 128 of them clean, exact or not, accepted or not, after
    // an own decision; 127 clean tiles or a HUD-less image lit on less than
    // half of the pixels leave the pair invalid (T1 reuses).
    for (const std::uint32_t matching : {127u, 128u})
      for (const std::uint32_t lit : {p / 2u - 1u, p / 2u})
        for (const std::uint32_t accepted : {0u, unsigned(candidate::hudless)})
          for (const std::uint32_t exact : {0u, unsigned(candidate::exact)}) {
            case_t test{"empty change set, " + std::to_string(matching) + " clean tiles, lit " + std::to_string(lit), {},
              candidate::hudless | exact, accepted};
            difference_rows(test.statistics, p, matching, 0, 900);
            test.statistics.lit[0][0] = lit;
            test.previous = selection::hold_state{detection::hold::own, 5u, 1234u};
            cases.push_back(test);
          }
    // The declared block: an accepted tag invalid this frame keeps accepted,
    // valid Backbuffer and current alpha out, and so does an accepted UIAlpha.
    {
      auto test = alpha_case("declared block, invalid accepted tag", p, candidate::ui_color | candidate::backbuffer | candidate::current,
        candidate::ui_color | candidate::backbuffer | candidate::current);
      alpha(test, k::ui_color, 0, p);
      alpha(test, k::backbuffer, 25000, 0);
      alpha(test, k::current, 25000, 0);
      cases.push_back(test);
      auto layer = alpha_case("declared block, invalid accepted UIAlpha beside an accepted layer", p,
        candidate::ui_alpha | candidate::layer, candidate::ui_alpha | candidate::layer);
      alpha(layer, k::ui_alpha, 10, 1000);
      alpha(layer, k::ui_layer, 2000, 0);
      cases.push_back(layer);
    }
    // An accepted, valid and empty UIAlpha decides empty.
    {
      auto test = alpha_case("accepted empty UIAlpha", p, candidate::ui_alpha | candidate::current, candidate::ui_alpha);
      alpha(test, k::current, 3000, 0);
      cases.push_back(test);
    }
    // An unaccepted opaque tag beside an accepted layer (SB HDR FG on).
    {
      auto test = alpha_case("unaccepted opaque tag beside an accepted layer", p,
        candidate::ui_color | candidate::layer | candidate::backbuffer, candidate::layer);
      alpha(test, k::ui_color, p, 0, p);
      alpha(test, k::ui_layer, 66, 0, 10);
      alpha(test, k::backbuffer, p, 0);
      cases.push_back(test);
    }
    // An invalid accepted layer (SB SDR scene image) neither decides nor blocks.
    {
      auto test = alpha_case("invalid accepted layer, accepted Backbuffer", p, candidate::layer | candidate::backbuffer,
        candidate::layer | candidate::backbuffer);
      alpha(test, k::ui_layer, 0, 30000);
      alpha(test, k::backbuffer, 95, 0);
      cases.push_back(test);
    }
    // Every H1 flag over each opacity claimant, accepted or not, beside a
    // HUD-less image changed on 90% of pixels (a pre-UI claim).
    for (const std::uint32_t offered : {unsigned(candidate::ui_alpha), unsigned(candidate::layer),
           unsigned(candidate::ui_alpha | candidate::layer), unsigned(candidate::ui_color)})
      for (const std::uint32_t accepted : {0u, offered})
        for (const std::uint32_t flags : {0u, hidden, pre_ui, hidden | pre_ui, hidden | refuted(offered),
               hidden | pre_ui | detection::per_frame_depth_not_current})
          for (const std::uint32_t invalid : {0u, 1u}) {
            auto test = alpha_case("H1 claim", p, offered | candidate::hudless, accepted, flags);
            alpha(test, k::ui_alpha, p, invalid, 36496);
            alpha(test, k::ui_color, p, invalid, 36496);
            alpha(test, k::ui_layer, p, invalid, 36495);
            difference_rows(test.statistics, p, 0, 900, 0);
            cases.push_back(test);
          }
    // Depth not current, counted on every frame.
    {
      auto test = alpha_case("depth not current", p, 0, 0, detection::per_frame_depth_not_current);
      cases.push_back(test);
    }
    // H1 at the opaque-full bound (0.99 p = 36495.36): an unaccepted layer
    // (b), an accepted Backbuffer and current beside an accepted partial
    // UIAlpha winner (the declared block keeps them out of S1 and of claim
    // (a)), an accepted UI color tag beside it (a), and an unaccepted
    // UIAlpha and UI color tag, which never claim.
    for (const std::uint32_t opaque : {36495u, 36496u})
      for (const std::uint32_t flags : {hidden, hidden | refuted(candidate::layer | candidate::backbuffer)}) {
        auto layer = alpha_case("H1 (b) unaccepted layer, " + std::to_string(opaque) + " opaque", p,
          candidate::layer | candidate::current, 0, flags);
        alpha(layer, k::ui_layer, p, 0, opaque);
        alpha(layer, k::current, p, 0, p);
        cases.push_back(layer);
        auto inferred = alpha_case("H1 blocked accepted Backbuffer and current beside a partial UIAlpha winner", p,
          candidate::ui_alpha | candidate::backbuffer | candidate::current,
          candidate::ui_alpha | candidate::backbuffer | candidate::current, flags);
        alpha(inferred, k::ui_alpha, 92, 0, 10);
        alpha(inferred, k::backbuffer, p, 0, opaque);
        alpha(inferred, k::current, p, 0, opaque);
        cases.push_back(inferred);
        auto tag = alpha_case("H1 (a) accepted UI color tag beside a partial UIAlpha winner", p,
          candidate::ui_alpha | candidate::ui_color, candidate::ui_alpha | candidate::ui_color, flags);
        alpha(tag, k::ui_alpha, 92, 0, 10);
        alpha(tag, k::ui_color, p, 0, opaque);
        cases.push_back(tag);
        auto declared = alpha_case("H1 unaccepted opaque UIAlpha and UI color tag", p,
          candidate::ui_alpha | candidate::ui_color, 0, flags);
        alpha(declared, k::ui_alpha, p, 0, opaque);
        alpha(declared, k::ui_color, p, 0, opaque);
        cases.push_back(declared);
        // An accepted opaque-full winner is overridden unless it is opaque on
        // every pixel.
        for (const std::uint32_t winner_opaque : {opaque, p}) {
          auto winner = alpha_case("H1 accepted opaque-full Backbuffer winner", p, candidate::backbuffer, candidate::backbuffer,
            flags);
          alpha(winner, k::backbuffer, p, 0, winner_opaque);
          cases.push_back(winner);
        }
      }
    // H1 (d): a layer without coverage whose signature is proven (fix 1),
    // V1-invalid or not, beside layers with coverage or without the proof,
    // with and without the pre-UI hold, the hidden hold and current depth;
    // and a HUD-less image changed on 90% of pixels or on fewer (0.9 p =
    // 33177.6). Texel 11 carries the layer's pre-UI pixel counts.
    const auto proven = detection::per_frame_pre_ui_proven;
    for (const std::uint32_t flags : {0u, hidden, pre_ui, hidden | pre_ui, hidden | pre_ui | detection::per_frame_depth_not_current,
           hidden | pre_ui | refuted(candidate::layer), proven, hidden | proven, pre_ui | proven, hidden | pre_ui | proven,
           hidden | pre_ui | proven | detection::per_frame_depth_not_current, hidden | pre_ui | proven | refuted(candidate::layer)})
      for (const std::uint32_t accepted : {0u, unsigned(candidate::layer)})
        for (const std::uint32_t covered : {0u, 1u, 2000u})
          for (const std::uint32_t invalid : {0u, 30000u}) {
            auto test = alpha_case("H1 (d) layer, " + std::to_string(covered) + " covered", p, candidate::layer | candidate::current,
              accepted, flags);
            alpha(test, k::ui_layer, covered, invalid, 0);
            alpha(test, k::current, p, 0, p);
            test.statistics.pre_ui[3] = {33200u, 33000u, 33100u, 3600u};
            cases.push_back(test);
            // Not a sample frame: no pre-UI threshold, the same decision and
            // a zero texel 11 whatever the tile rows hold.
            test.name += ", no pre-UI threshold";
            test.pre_ui_threshold = 0.f;
            cases.push_back(test);
          }
    for (const std::uint32_t flags : {0u, hidden, pre_ui, hidden | pre_ui, hidden | pre_ui | detection::per_frame_depth_not_current,
           hidden | pre_ui | refuted(candidate::layer), hidden | pre_ui | proven})
      for (const std::uint32_t accepted : {0u, unsigned(candidate::layer)}) {
        for (const std::uint32_t changed : {33177u, 33178u}) {
          case_t hudless{"H1 (d) HUD-less image, " + std::to_string(changed) + " changed", {},
            candidate::hudless | candidate::layer, accepted, flags};
          difference_rows(hudless.statistics, p, 0, 0, 0);
          hudless.statistics.difference[0][0] = changed;
          alpha(hudless, k::ui_layer, 0, 30000, 0);
          cases.push_back(hudless);
        }
      }
    // H1 over a stored decision: a frame without its own decision reuses a
    // stored 8 once, and an H1 frame stores its own 8.
    {
      auto test = alpha_case("H1 stored 8 reused", p, candidate::backbuffer, candidate::backbuffer, hidden);
      alpha(test, k::backbuffer, 400, 2000);
      test.previous = selection::hold_state{detection::hold::own, 8u, p};
      cases.push_back(test);
      auto stored = alpha_case("H1 decides and stores 8", p, candidate::layer | candidate::backbuffer, candidate::backbuffer,
        hidden);
      alpha(stored, k::ui_layer, p, 0, p);
      alpha(stored, k::backbuffer, 400, 2000);
      stored.previous = selection::hold_state{detection::hold::own, 3u, 77u};
      cases.push_back(stored);
    }
    // A2: an accepted full Backbuffer over a valid exact pair, contradicted by
    // a tenth of its strong pixels or just not; the same over an inexact pair
    // or a middle-band pair, which never judge; and the layer and current
    // alpha judged together.
    for (const std::uint32_t contradicted : {3686u, 3687u, 0u})
      for (const std::uint32_t pair : {unsigned(candidate::hudless | candidate::exact), unsigned(candidate::hudless)})
        for (const bool middle : {false, true})
          for (const std::uint32_t accepted : {unsigned(candidate::backbuffer), unsigned(candidate::backbuffer | candidate::layer)}) {
            auto test = alpha_case("one-way judgment, " + std::to_string(contradicted) + " contradicted", p,
              candidate::backbuffer | candidate::layer | candidate::current | pair, accepted);
            difference_rows(test.statistics, p, middle ? 0 : 128, 100, middle ? 500 : 900);
            alpha(test, k::backbuffer, p, 0);
            alpha(test, k::ui_layer, 3000, 0);
            alpha(test, k::current, 2000, 0);
            judgment(test, k::backbuffer, 36864, contradicted);
            judgment(test, k::ui_layer, 2000, 200);
            judgment(test, k::current, 1000, 99);
            cases.push_back(test);
          }
    // T1: an accepted Backbuffer invalid this frame has no decision of its
    // own: it reuses the previous own decision once, then has no mask; a
    // reset never reuses, and a frame that decides itself stores own.
    const selection::hold_state own_full{detection::hold::own, 3u, p}, own_partial{detection::hold::own, 10u, 900u};
    {
      auto test = alpha_case("T1 grace, invalid accepted Backbuffer", p, candidate::backbuffer, candidate::backbuffer);
      alpha(test, k::backbuffer, 400, 2000);
      test.previous = own_partial;
      cases.push_back(test);
      test.name = "T1 grace spent";
      test.previous = std::nullopt;
      cases.push_back(test);
      test.name = "T1 reset";
      test.previous = own_partial;
      test.flags = detection::per_frame_hold_reset;
      cases.push_back(test);
      auto decided = alpha_case("T1 own decision after a spent grace", p, candidate::backbuffer, candidate::backbuffer);
      alpha(decided, k::backbuffer, 400, 0);
      decided.previous = selection::hold_state{detection::hold::spent, 3u, 5u};
      cases.push_back(decided);
    }
    // T1 with zero offers: an accepted candidate is missing (E33 FG on), so
    // the previous full Backbuffer applies once, a whole-frame alpha.
    for (const std::uint32_t reset : {0u, unsigned(detection::per_frame_hold_reset)}) {
      auto test = alpha_case("T1 accepted missing, zero offers", p, 0, candidate::backbuffer,
        detection::per_frame_accepted_missing | reset);
      test.previous = own_full;
      cases.push_back(test);
    }
    // An unaccepted candidate missing is not T1: decided frames store own.
    {
      auto test = alpha_case("T1 nothing accepted", p, candidate::current, 0);
      alpha(test, k::current, 400, 0);
      test.previous = own_full;
      cases.push_back(test);
    }
    // F1: every no-mask reason with its refused candidate.
    {
      auto blocked = alpha_case("F1 presented_blocked", p, candidate::ui_color | candidate::layer | candidate::current,
        candidate::ui_color | candidate::layer | candidate::current);
      alpha(blocked, k::ui_color, 0, p);
      alpha(blocked, k::current, 300, 0);
      alpha(blocked, k::ui_layer, 300, 0);
      cases.push_back(blocked);
      auto invalid = alpha_case("F1 trusted_invalid", p, candidate::current | candidate::backbuffer, candidate::current |
        candidate::backbuffer);
      alpha(invalid, k::current, 0, 1000);
      alpha(invalid, k::backbuffer, 0, 1000);
      cases.push_back(invalid);
      auto aside = alpha_case("F1 layer_aside", p, candidate::layer | candidate::current, 0);
      alpha(aside, k::ui_layer, 0, 1000);
      cases.push_back(aside);
      auto unaccepted = alpha_case("F1 unaccepted", p, candidate::ui_color | candidate::backbuffer, candidate::ui_alpha);
      alpha(unaccepted, k::ui_color, 0, 0);
      alpha(unaccepted, k::backbuffer, 300, 0);
      cases.push_back(unaccepted);
      case_t failed{"F1 difference_failed", {}, candidate::hudless, candidate::hudless};
      difference_rows(failed.statistics, p, 0, 500, 500);
      cases.push_back(failed);
      auto ambiguous = alpha_case("F1 ambiguous", p, candidate::backbuffer | candidate::current, 0);
      alpha(ambiguous, k::backbuffer, p, 0);
      cases.push_back(ambiguous);
      auto other = alpha_case("F1 other", p, candidate::backbuffer, 0);
      alpha(other, k::backbuffer, 0, 1000);
      cases.push_back(other);
    }
    // H2 (fix 2): a frame applying no source that T1 did not reuse shows flat
    // (11) under still::flatten; the hold store keeps the decision before it.
    {
      const auto flatten = detection::still::flatten;
      // Stellar Blade SDR loading: the bare cleared layer (V1-invalid) and
      // the unaccepted current alpha decide nothing.
      auto loading = alpha_case("H2 still screen without a UI source", p, candidate::layer | candidate::current, 0);
      alpha(loading, k::ui_layer, 0, 2000);
      loading.still = flatten;
      cases.push_back(loading);
      loading.name = "H2 shadow (no flag)";
      loading.still = 0;
      cases.push_back(loading);
      loading.name = "H2 other b2 word 5 bits";
      loading.still = ~flatten;
      cases.push_back(loading);
      // E33 SDR: an accepted current alpha deciding an empty mask is respected.
      auto empty = alpha_case("H2 accepted empty decision", p, candidate::current, candidate::current);
      empty.still = flatten;
      cases.push_back(empty);
      // H1 wins over H2.
      auto h1 = alpha_case("H2 under H1", p, candidate::layer, 0, hidden);
      alpha(h1, k::ui_layer, p, 0, p);
      h1.still = flatten;
      cases.push_back(h1);
      // A T1-reused decision, even a stored no-mask one, is never H2's.
      auto reused = alpha_case("H2 on a reused decision", p, candidate::backbuffer, candidate::backbuffer);
      alpha(reused, k::backbuffer, 400, 2000);
      reused.previous = selection::hold_state{detection::hold::own, 0u, 0u};
      reused.still = flatten;
      cases.push_back(reused);
      reused.name = "H2 on a reused source";
      reused.previous = own_partial;
      cases.push_back(reused);
      // Reused depth does not stop H2 (unlike H1).
      auto stale = alpha_case("H2 on reused depth", p, candidate::layer | candidate::current, 0,
        detection::per_frame_depth_not_current);
      alpha(stale, k::ui_layer, 0, 2000);
      stale.still = flatten;
      cases.push_back(stale);
      // The next real frame finds the hold store the H2 frame wrote: its
      // own decision (none), not 11.
      auto chained = alpha_case("H2 then an invalid accepted Backbuffer", p, candidate::backbuffer, candidate::backbuffer);
      alpha(chained, k::backbuffer, 400, 2000);
      auto first = loading;
      first.name = "H2 before a chained frame";
      first.still = flatten;
      cases.push_back(first);
      chained.previous = std::nullopt;
      cases.push_back(chained);
    }
    for (auto &test : identity_cases()) cases.push_back(std::move(test));
    return cases;
  }

  // Values near every bound of a total p.
  std::uint32_t pick(std::mt19937 &random, std::uint32_t p) {
    switch (std::uniform_int_distribution<int>(0, 11)(random)) {
      case 0: return 0;
      case 1: return 1;
      case 2: return p / 100u;
      case 3: return p / 100u + 1u;
      case 4: return (p * 9u + 9u) / 10u - 1u;
      case 5: return (p * 9u + 9u) / 10u;
      case 6: return (p * 99u + 99u) / 100u - 1u;
      case 7: return (p * 99u + 99u) / 100u;
      case 8: return p / 2u;
      case 9: return p;
      default: return std::uniform_int_distribution<std::uint32_t>(0, p)(random);
    }
  }

  case_t random_case(std::mt19937 &random, unsigned index) {
    static constexpr std::array<std::uint32_t, 4> sizes{36864u, 921600u, 8294400u, 14745600u};
    const std::uint32_t p = sizes[std::uniform_int_distribution<std::size_t>(0, sizes.size() - 1)(random)];
    case_t test{"random case " + std::to_string(index), {}, 0, 0, 0};
    test.offered = std::uniform_int_distribution<std::uint32_t>(0, 0x7f)(random);
    switch (std::uniform_int_distribution<int>(0, 3)(random)) {
      case 0: test.accepted = 0; break;
      case 1: test.accepted = test.offered & selection::candidate_bits; break;
      default: test.accepted = std::uniform_int_distribution<std::uint32_t>(0, 0x7f)(random) & selection::candidate_bits;
    }
    // Every per-frame bit, the retired ones (0x10000, 0x80000, 0x20000000)
    // included, and refuted candidate bits in bits 24-30.
    static constexpr std::array<std::uint32_t, 9> per_frame{detection::per_frame_scene_hidden,
      detection::per_frame_pre_ui_visible, detection::per_frame_depth_not_current, detection::per_frame_accepted_missing,
      detection::per_frame_hold_reset, detection::per_frame_pre_ui_proven, 0x10000u, 0x80000u, 0x20000000u};
    for (const auto bit : per_frame)
      if (random() & 1u) test.flags |= bit;
    if (random() & 1u) test.flags |= (std::uint32_t(random()) & 0x7fu) << detection::per_frame_refuted_shift;
    test.flags |= random() & 7u;
    // The previous real frame's hold store, or (one case in eight) whatever
    // the previous case's reduce wrote.
    if (random() % 8u) {
      const std::uint32_t state = std::uniform_int_distribution<std::uint32_t>(0, 3)(random);
      const std::uint32_t source = std::uniform_int_distribution<std::uint32_t>(0, 10)(random);
      test.previous = selection::hold_state{state == 3 ? std::uint32_t(random()) : state, source, pick(random, p)};
    } else {
      test.previous = std::nullopt;
    }
    auto &t = test.statistics;
    for (std::size_t k = 0; k != 4; ++k) {
      spread(t.coverage, k, pick(random, p), random);
      spread(t.invalid, k, pick(random, p) / ((random() & 3u) ? 64u : 1u), random);
    }
    for (std::size_t k = 0; k != 4; ++k) spread(t.layer, k, k == 1 ? pick(random, p) / ((random() & 3u) ? 64u : 1u) :
      pick(random, p), random);
    for (std::size_t k = 0; k != 4; ++k) spread(t.lit, k, pick(random, p), random);
    for (std::size_t k = 0; k != 4; ++k) spread(t.pre_ui, k, pick(random, p), random);
    // One-way counts near the tenth bound.
    for (std::size_t k = 0; k != 3; ++k) {
      const std::uint32_t strong = pick(random, p);
      std::uint32_t contradicted = 0;
      switch (std::uniform_int_distribution<int>(0, 4)(random)) {
        case 0: contradicted = strong / 10u; break;
        case 1: contradicted = (strong + 9u) / 10u; break;
        case 2: contradicted = (strong + 9u) / 10u ? (strong + 9u) / 10u - 1u : 0u; break;
        case 3: contradicted = std::uniform_int_distribution<std::uint32_t>(0, strong)(random); break;
        default: contradicted = pick(random, p);
      }
      spread(t.strong, k, strong, random);
      spread(t.contradicted, k, contradicted, random);
    }
    // The difference rows decide the matching tiles, so they keep per-tile
    // structure: a matching share of clean tiles and the rest at one ratio.
    const std::uint32_t matching = std::uniform_int_distribution<std::uint32_t>(0, 3)(random) == 0 ?
      std::uniform_int_distribution<std::uint32_t>(126, 130)(random) : std::uniform_int_distribution<std::uint32_t>(0, 256)(random);
    switch (std::uniform_int_distribution<int>(0, 3)(random)) {
      case 0: difference_rows(t, p, 0, std::uniform_int_distribution<std::uint32_t>(970, 1000)(random), 0); break;
      case 1: difference_rows(t, p, matching, 100, 900); break;
      case 2: difference_rows(t, p, matching, std::uniform_int_distribution<std::uint32_t>(0, 300)(random),
        std::uniform_int_distribution<std::uint32_t>(400, 989)(random)); break;
      default: difference_rows(t, p, matching, std::uniform_int_distribution<std::uint32_t>(0, 1000)(random), 0);
    }
    if (!(random() % 8u)) t.difference[random() % 256u][1] = 1u;
    // H2's flag in half the cases, other bits of b2 word 5 now and then.
    if (random() & 1u) test.still = detection::still::flatten | ((random() % 8u) ? 0u : std::uint32_t(random()) & ~1u);
    // S3: stamps and proposals near each other in half the cases, the token
    // batch and the gate (and the reserved bit 0x2) in a quarter.
    if (random() & 1u) {
      const auto label = [&] { return std::uniform_int_distribution<std::uint32_t>(0u, 3u)(random); };
      test.identity = {label(), label(), label(), label(), label(), label(), 0u};
      if (!(random() % 16u)) test.identity.layer_present = 0xfffffff0u + label();
      if (!(random() % 4u)) test.identity.bits = std::uint32_t(random()) & 7u;
    }
    return test;
  }

  // H2's stillness counts (fix 2): the scene compare pass compares every
  // cell's presented-luma mean (the cell sums at t10) with the previous
  // measured sample's (u5: 0 none, else 0x80000000 | mean), counts compared
  // and still cells (within still::tolerance) per 16x16-cell group into
  // statistics rows 144-152 and stores the new means; the evidence pass sums
  // them into decision texel 12. Inactive depth writes nothing and texel 12
  // zero. Compared with a CPU oracle at the compiled 256 x 144 (one pixel per
  // cell), sRGB.
  void check_stillness(gpu_t &gpu, const std::string &source) {
    namespace scene = detection::scene;
    constexpr std::uint32_t cells_x = scene::cells_x, cells_y = scene::cells_y, cells = cells_x * cells_y;
    const auto compare = compile_pass(gpu, source, 1, "SunshineSceneCompareCS");
    const auto evidence = compile_pass(gpu, source, 1, "SunshineSceneEvidenceCS");
    const auto texture = [&](UINT width, UINT height, DXGI_FORMAT format, UINT bind, const void *initial, UINT pitch,
                           ComPtr<ID3D11Texture2D> &result, ComPtr<ID3D11Texture2D> &staging) {
      D3D11_TEXTURE2D_DESC desc{};
      desc.Width = width;
      desc.Height = height;
      desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
      desc.Format = format;
      desc.BindFlags = bind;
      const D3D11_SUBRESOURCE_DATA data{initial, pitch, 0};
      checked(gpu.device->CreateTexture2D(&desc, initial ? &data : nullptr, &result), "stillness texture");
      desc.BindFlags = 0;
      desc.Usage = D3D11_USAGE_STAGING;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      checked(gpu.device->CreateTexture2D(&desc, nullptr, &staging), "stillness staging");
    };
    // Cell sums {presented luma, pre-UI luma, parallax, pixels}: one pixel
    // per cell, so a cell's mean is its luma; no parallax, so no edge cell.
    std::vector<texel> sums(cells);
    std::vector<std::uint32_t> means(cells), previous(cells);
    std::mt19937 random(0x5711u);
    std::uint32_t expected_compared = 0, expected_still = 0;
    std::vector<texel> expected_rows(16 * 9);
    for (std::uint32_t i = 0; i != cells; ++i) {
      const std::uint32_t mean = std::uniform_int_distribution<std::uint32_t>(10000u, scene::luma_scale - 10000u)(random);
      means[i] = mean;
      sums[i] = {mean, 0u, 0u, 1u};
      // none, equal, +tolerance, -(tolerance + 1), -tolerance, +(tolerance + 1)
      const auto t = detection::still::tolerance;
      std::uint32_t stored = 0;
      switch (i % 6u) {
        case 0: stored = 0u; break;
        case 1: stored = mean; break;
        case 2: stored = mean + t; break;
        case 3: stored = mean - t - 1u; break;
        case 4: stored = mean - t; break;
        default: stored = mean + t + 1u;
      }
      previous[i] = i % 6u ? 0x80000000u | stored : 0u;
      const bool compared = i % 6u != 0u, still = compared && (i % 6u == 1u || i % 6u == 2u || i % 6u == 4u);
      const std::uint32_t x = i % cells_x, y = i / cells_x, group = (y / 16u) * 16u + x / 16u;
      expected_rows[group][0] += still;
      expected_rows[group][1] += compared;
      expected_compared += compared;
      expected_still += still;
    }
    ComPtr<ID3D11Texture2D> cell_sums, cell_staging, previous_store, previous_staging, statistics, statistics_staging, decision,
      decision_staging;
    texture(cells_x, cells_y, DXGI_FORMAT_R32G32B32A32_UINT, D3D11_BIND_SHADER_RESOURCE, sums.data(), cells_x * sizeof(texel),
      cell_sums, cell_staging);
    texture(cells_x, cells_y, DXGI_FORMAT_R32_UINT, D3D11_BIND_UNORDERED_ACCESS, previous.data(), cells_x * sizeof(std::uint32_t),
      previous_store, previous_staging);
    const std::uint32_t rows = detection::statistics_rows(detection::max_scene_evidence_images);
    const std::vector<texel> sentinel_rows(16 * rows, texel{0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu});
    texture(16, rows, DXGI_FORMAT_R32G32B32A32_UINT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
      sentinel_rows.data(), 16 * sizeof(texel), statistics, statistics_staging);
    texture(detection::still_decision_texels, 1, DXGI_FORMAT_R32G32B32A32_UINT, D3D11_BIND_UNORDERED_ACCESS, nullptr, 0, decision,
      decision_staging);
    ComPtr<ID3D11ShaderResourceView> cells_view, statistics_view;
    ComPtr<ID3D11UnorderedAccessView> previous_view, statistics_uav, decision_view;
    checked(gpu.device->CreateShaderResourceView(cell_sums.Get(), nullptr, &cells_view), "cells view");
    checked(gpu.device->CreateShaderResourceView(statistics.Get(), nullptr, &statistics_view), "statistics view");
    checked(gpu.device->CreateUnorderedAccessView(previous_store.Get(), nullptr, &previous_view), "previous view");
    checked(gpu.device->CreateUnorderedAccessView(statistics.Get(), nullptr, &statistics_uav), "statistics uav");
    checked(gpu.device->CreateUnorderedAccessView(decision.Get(), nullptr, &decision_view), "decision uav");
    // b0: depth ready with a valid camera (SunshineSceneDepthActive), or not.
    struct geometry {
      float depth_adjustment;
      std::int32_t depth_map_view;
      std::uint32_t depth_ready, camera_ready;
      std::int32_t basis;
      float depth_scale, strength_blend, disparity_limit;
      float projection[2], raw_range[2], convergence[2], jitter[2], rect[4];
    };
    static_assert(sizeof(geometry) == 80);
    D3D11_BUFFER_DESC buffer{};
    buffer.ByteWidth = 80;
    buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    ComPtr<ID3D11Buffer> b0;
    checked(gpu.device->CreateBuffer(&buffer, nullptr, &b0), "b0");
    // S3: the evidence pass rewrites texel 12 .z/.w with the identity words
    // the reduce wrote: an offered layer pair one Present late.
    const selection::identity_input stamped{42u, 7u, 0u, 41u, 0u, 0u};
    const auto constants = with_identity_words({candidate::layer, 2.f / 255.f, 0u, 0u, 0.f, 0u, {}}, stamped);
    const auto verdict = selection::verify_identity(stamped, candidate::layer);
    require(verdict.layer == identity::mismatch && verdict.layer_delta == 1, "S3: the evidence pass's case is not a late copy");
    gpu.context->UpdateSubresource(gpu.constants.Get(), 0, nullptr, &constants, 0, 0);
    upload_stamps(gpu, stamped);
    const auto pass = [&](bool active) {
      const geometry g{1.f, 0, active ? 1u : 0u, active ? 1u : 0u, 0, 1.f, 1.f, .02f, {0.f, 1.f}, {0.f, 0.f}, {.5f, .1f},
        {0.f, 0.f}, {0.f, 0.f, 1.f, 1.f}};
      gpu.context->UpdateSubresource(b0.Get(), 0, nullptr, &g, 0, 0);
      const std::array<std::uint32_t, 4> sentinel{0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
      gpu.context->ClearUnorderedAccessViewUint(decision_view.Get(), sentinel.data());
      ID3D11Buffer *buffers[3]{b0.Get(), nullptr, gpu.constants.Get()};
      gpu.context->CSSetConstantBuffers(0, 3, buffers);
      // Compare: cell sums at t10, the previous means at u5, statistics at u6.
      ID3D11ShaderResourceView *views[11]{};
      views[10] = cells_view.Get();
      gpu.context->CSSetShader(compare.Get(), nullptr, 0);
      gpu.context->CSSetShaderResources(0, 11, views);
      ID3D11UnorderedAccessView *uavs[2]{previous_view.Get(), statistics_uav.Get()};
      gpu.context->CSSetUnorderedAccessViews(5, 2, uavs, nullptr);
      gpu.context->Dispatch(cells_x / 16, cells_y / 16, 1);
      ID3D11UnorderedAccessView *none[2]{};
      gpu.context->CSSetUnorderedAccessViews(5, 2, none, nullptr);
      // Evidence: statistics at t10, the stamps at t9, decision texels at u6.
      views[9] = gpu.stamps_view.Get();
      views[10] = statistics_view.Get();
      gpu.context->CSSetShader(evidence.Get(), nullptr, 0);
      gpu.context->CSSetShaderResources(0, 11, views);
      ID3D11UnorderedAccessView *decision_uav = decision_view.Get();
      gpu.context->CSSetUnorderedAccessViews(6, 1, &decision_uav, nullptr);
      gpu.context->Dispatch(1, 1, 1);
      gpu.context->CSSetUnorderedAccessViews(6, 1, none, nullptr);
      ID3D11ShaderResourceView *cleared[11]{};
      gpu.context->CSSetShaderResources(0, 11, cleared);
    };
    namespace word = detection::decision_word;
    // Inactive depth: nothing stored or counted, texel 12 zero.
    pass(false);
    auto stored = read<std::uint32_t>(gpu, previous_store.Get(), previous_staging.Get(), cells);
    auto statistics_rows = read<texel>(gpu, statistics.Get(), statistics_staging.Get(), 16 * rows);
    auto words = read<std::uint32_t>(gpu, decision.Get(), decision_staging.Get(), 4 * detection::still_decision_texels);
    require(stored == previous && statistics_rows[16 * detection::still_statistics_row][0] == 0xdeadbeefu &&
        !words[word::still_cells] && !words[word::still_compared] &&
        words[word::id_verdicts] == selection::identity_verdict_word(verdict) &&
        words[word::id_deltas] == selection::identity_delta_word(verdict),
      "H2 stillness: inactive depth stored or counted cells, or texel 12 is not zero with S3's identity words");
    // Active: per-group counts, the stored means and texel 12.
    pass(true);
    stored = read<std::uint32_t>(gpu, previous_store.Get(), previous_staging.Get(), cells);
    statistics_rows = read<texel>(gpu, statistics.Get(), statistics_staging.Get(), 16 * rows);
    words = read<std::uint32_t>(gpu, decision.Get(), decision_staging.Get(), 4 * detection::still_decision_texels);
    for (std::uint32_t i = 0; i != cells; ++i)
      if (stored[i] != (0x80000000u | means[i]))
        throw std::runtime_error("H2 stillness: cell " + std::to_string(i) + " stored " + std::to_string(stored[i]) +
          ", not its mean with the stored bit");
    for (std::uint32_t group = 0; group != 16u * 9u; ++group) {
      const auto &got = statistics_rows[(detection::still_statistics_row + group / 16u) * 16u + group % 16u];
      if (got[0] != expected_rows[group][0] || got[1] != expected_rows[group][1] || got[2] || got[3])
        throw std::runtime_error("H2 stillness: group " + std::to_string(group) + " counted " + std::to_string(got[0]) + '/' +
          std::to_string(got[1]) + ", the CPU oracle " + std::to_string(expected_rows[group][0]) + '/' +
          std::to_string(expected_rows[group][1]));
    }
    require(words[word::still_cells] == expected_still && words[word::still_compared] == expected_compared &&
        words[word::id_verdicts] == selection::identity_verdict_word(verdict) &&
        words[word::id_deltas] == selection::identity_delta_word(verdict) && expected_still * 5u == expected_compared * 3u &&
        expected_compared * 6u == cells * 5u, "H2 stillness: texel 12 does not sum the groups");
    // The next sample of the same image: every cell compared and still.
    pass(true);
    words = read<std::uint32_t>(gpu, decision.Get(), decision_staging.Get(), 4 * detection::still_decision_texels);
    require(words[word::still_cells] == cells && words[word::still_compared] == cells,
      "H2 stillness: an unchanged image did not read every cell still");
    // A uniform change of exactly the tolerance is still; one more is not.
    for (const std::uint32_t delta : {detection::still::tolerance, detection::still::tolerance + 1u}) {
      for (std::uint32_t i = 0; i != cells; ++i) sums[i][0] = means[i] + (delta == detection::still::tolerance ? delta : 0u);
      if (delta != detection::still::tolerance)
        for (std::uint32_t i = 0; i != cells; ++i) sums[i][0] = means[i] + detection::still::tolerance + delta;
      gpu.context->UpdateSubresource(cell_sums.Get(), 0, nullptr, sums.data(), cells_x * sizeof(texel), 0);
      pass(true);
      words = read<std::uint32_t>(gpu, decision.Get(), decision_staging.Get(), 4 * detection::still_decision_texels);
      const bool still = delta == detection::still::tolerance;
      require(words[word::still_compared] == cells && words[word::still_cells] == (still ? cells : 0u),
        still ? "H2 stillness: a change of exactly the tolerance was not still" :
                "H2 stillness: a change beyond the tolerance was still");
    }
  }

  ComPtr<ID3D11ShaderResourceView> float_image(gpu_t &gpu, const std::vector<std::array<float, 4>> &texels) {
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = 256;
    desc.Height = 144;
    desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    const D3D11_SUBRESOURCE_DATA data{texels.data(), 256 * sizeof(texels[0]), 0};
    ComPtr<ID3D11Texture2D> texture;
    checked(gpu.device->CreateTexture2D(&desc, &data, &texture), "image");
    ComPtr<ID3D11ShaderResourceView> view;
    checked(gpu.device->CreateShaderResourceView(texture.Get(), nullptr, &view), "image view");
    return view;
  }

  // The tiles pass's statistics (rows 0-111: alpha coverage and invalid
  // pixels, the HUD-less difference, lit and opaque pixels, the layer's
  // counts and the one-way judgment counts) on images of edge values, against
  // a CPU count of the rules: covered is in (0, 1], invalid is non-finite or
  // out of range, opaque at least 254/255 (every alpha kind: rows 48 .yzw and
  // 64 .zw), and the layer alone also fails
  // beyond the premultiplied bound of its flags (V1); changed beyond the
  // threshold, unchanged within half of it, lit beyond eight times it (V2);
  // strong is alpha in [1/2, 1], contradicted a strong pixel where an offered
  // exact pair's HUD-less image is lit and unchanged, neither for the late
  // layer (A2, E2); and rows 128-143, the layer's colour against the
  // presented colour (t6) at 8 times pre_ui_threshold (b2 word 4; zero
  // compares nothing): matching, lit layer, lit presented and lit presented
  // but different pixels, relative above one in scRGB (H1 d).
  void check_tiles(gpu_t &gpu, ID3D11ComputeShader *tiles, std::uint32_t flags, std::uint32_t offered, const char *space,
      unsigned color, float pre_ui_threshold) {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const std::array<float, 15> alphas{0.f, -0.f, 1e-30f, .5f, 1.f, 1.5f, -.5f, nan, inf, .999f, .99f, .25f, .49999997f, -inf,
      254.f / 255.f};
    const std::array<std::array<float, 4>, 12> layers{{{0.f, 0.f, 0.f, 0.f}, {.5f, .5f, .5f, .5f}, {1.f, 1.f, 1.f, .1f},
      {.3f, 0.f, 0.f, 0.f}, {nan, 0.f, 0.f, .2f}, {0.f, 0.f, 0.f, -0.f}, {20.f, 0.f, 0.f, .5f}, {.2f, .2f, .2f, .999f},
      {0.f, 0.f, 0.f, nan}, {0.f, 0.f, 0.f, inf}, {.1f, .1f, .1f, 1.f}, {0.f, 0.f, 0.f, .49999997f}}};
    // The current color (t0) and HUD-less (t14) rgb: lit or dark, equal,
    // changed or non-finite.
    const std::array<float, 4> colors{.5f, .01f, 0.f, .25f};
    const std::array<float, 4> offsets{0.f, .5f, nan, .001f};
    // The presented colour (t6): the layer's rgb moved by these multiples of
    // the coarse threshold (none on a bound), or non-finite, or far brighter.
    const float coarse = pre_ui_threshold * 8.f;
    const std::array<float, 7> presented_offsets{0.f, .25f, -.5f, 3.f, -2.f, nan, 40.f};
    std::array<std::vector<std::array<float, 4>>, 7> images; // t11, t12, t13, t0, t7, t14, t6
    for (auto &image : images) image.resize(256 * 144);
    for (std::uint32_t y = 0; y != 144; ++y)
      for (std::uint32_t x = 0; x != 256; ++x) {
        const auto i = y * 256 + x;
        const float color = colors[(x * 5 + y * 3) % colors.size()], offset = offsets[(x + y * 7) % offsets.size()];
        images[0][i] = {alphas[(x + y) % alphas.size()], 0.f, 0.f, 1.f};
        images[1][i] = {0.f, 0.f, 0.f, alphas[(x * 3 + y) % alphas.size()]};
        images[2][i] = {0.f, 0.f, 0.f, alphas[(x + 5 * y) % alphas.size()]};
        images[3][i] = {color, color, color, alphas[(x * 7 + y * 2) % alphas.size()]};
        images[4][i] = layers[(x + 2 * y) % layers.size()];
        images[5][i] = {color + offset, color, color, 1.f};
        const float moved = presented_offsets[(x * 3 + y * 5) % presented_offsets.size()] *
          (pre_ui_threshold > 0.f ? coarse : 16.f / 255.f);
        const auto &l = images[4][i];
        images[6][i] = {l[0] + moved, l[1], l[2] + moved * .5f, 1.f};
      }
    // A candidate that is not offered is not bound, as the renderer and the
    // replay bind them: it reads zero (the pass skips its loads).
    const std::array<std::pair<std::size_t, std::uint32_t>, 5> slots{{{0, candidate::ui_alpha}, {1, candidate::ui_color},
      {2, candidate::backbuffer}, {4, candidate::layer}, {5, candidate::hudless}}};
    for (const auto &[image, bit] : slots)
      if (!(offered & bit)) std::fill(images[image].begin(), images[image].end(), std::array<float, 4>{});
    const auto okay = [](float a) { return std::isfinite(a) && a >= 0.f && a <= 1.f; };
    const auto strong = [](float a) { return a >= .5f && a <= 1.f; };
    const float opaque = 254.f / 255.f, headroom = (flags & detection::stored_hdr_headroom) ? 125.f : 2.f;
    const float threshold = 2.f / 255.f;
    const bool exact = (offered & (candidate::hudless | candidate::exact)) == (candidate::hudless | candidate::exact);
    const bool late = (flags & detection::stored_late_layer) != 0u;
    const auto peak = [](const std::array<float, 4> &v) { return std::max({std::abs(v[0]), std::abs(v[1]), std::abs(v[2])}); };
    std::array<texel, 256> coverage{}, invalid{}, difference{}, lit{}, layer{}, strong_counts{}, contradicted{}, pre_ui{};
    for (std::uint32_t y = 0; y != 144; ++y)
      for (std::uint32_t x = 0; x != 256; ++x) {
        const auto i = y * 256 + x, tile = (y / 9) * 16 + x / 16;
        const float a[4]{images[0][i][0], images[1][i][3], images[2][i][3], images[3][i][3]};
        for (std::size_t k = 0; k != 4; ++k) {
          coverage[tile][k] += okay(a[k]) && a[k] > 0.f;
          invalid[tile][k] += !okay(a[k]);
        }
        lit[tile][1] += okay(a[0]) && a[0] >= opaque;
        lit[tile][2] += okay(a[1]) && a[1] >= opaque;
        lit[tile][3] += okay(a[2]) && a[2] >= opaque;
        const auto &l = images[4][i];
        const bool bound = (flags & detection::stored_premultiplied) && std::max({l[0], l[1], l[2]}) > l[3] * headroom + 4.f / 255.f;
        layer[tile][0] += okay(l[3]) && l[3] > 0.f;
        layer[tile][1] += !okay(l[3]) || bound;
        layer[tile][2] += okay(l[3]) && l[3] >= opaque;
        layer[tile][3] += okay(a[3]) && a[3] >= opaque;
        // SunshineHUDlessDifference, relative above one in scRGB.
        const auto &current = images[3][i], &hudless = images[5][i];
        const bool finite = std::isfinite(current[0]) && std::isfinite(current[1]) && std::isfinite(current[2]) &&
          std::isfinite(hudless[0]) && std::isfinite(hudless[1]) && std::isfinite(hudless[2]);
        const float delta = std::max({std::abs(current[0] - hudless[0]), std::abs(current[1] - hudless[1]),
          std::abs(current[2] - hudless[2])}) / (color == 2 ? std::max(1.f, peak(current)) : 1.f);
        const bool unchanged = finite && delta <= threshold * .5f;
        const bool lit_pixel = finite && std::max({std::abs(hudless[0]), std::abs(hudless[1]), std::abs(hudless[2])}) > threshold * 8.f;
        difference[tile][0] += finite && delta > threshold;
        difference[tile][1] += !finite;
        difference[tile][2] += unchanged;
        difference[tile][3] += 1u;
        lit[tile][0] += lit_pixel;
        const bool judged[3]{!late && strong(l[3]), strong(a[2]), strong(a[3])};
        const bool shown = exact && lit_pixel && unchanged;
        for (std::size_t k = 0; k != 3; ++k) {
          strong_counts[tile][k] += judged[k];
          contradicted[tile][k] += judged[k] && shown;
        }
        // H1 (d): the layer's colour against the presented colour.
        const auto &presented = images[6][i];
        const bool compared = coarse > 0.f && std::isfinite(presented[0]) && std::isfinite(presented[1]) &&
          std::isfinite(presented[2]) && std::isfinite(l[0]) && std::isfinite(l[1]) && std::isfinite(l[2]);
        if (compared) {
          const float scale = color == 2 ? std::max(1.f, peak(presented)) : 1.f;
          const float apart = std::max({std::abs(presented[0] - l[0]), std::abs(presented[1] - l[1]),
            std::abs(presented[2] - l[2])}) / scale;
          const bool presented_lit = peak(presented) > coarse;
          pre_ui[tile][0] += apart <= coarse;
          pre_ui[tile][1] += peak(l) > coarse;
          pre_ui[tile][2] += presented_lit;
          pre_ui[tile][3] += presented_lit && apart > coarse;
        }
      }
    std::array<ComPtr<ID3D11ShaderResourceView>, 7> views;
    for (std::size_t k = 0; k != views.size(); ++k) views[k] = float_image(gpu, images[k]);
    D3D11_TEXTURE2D_DESC desc{};
    constexpr std::size_t columns = detection::statistics_columns(detection::tile_parts);
    desc.Width = UINT(columns);
    desc.Height = detection::statistics_rows(0);
    desc.MipLevels = desc.ArraySize = desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    ComPtr<ID3D11Texture2D> statistics, staging;
    ComPtr<ID3D11UnorderedAccessView> statistics_view;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &statistics), "tile statistics");
    checked(gpu.device->CreateUnorderedAccessView(statistics.Get(), nullptr, &statistics_view), "tile statistics view");
    desc.BindFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &staging), "tile statistics staging");
    const detection_constants constants{offered, threshold, 0u, flags, pre_ui_threshold, {}};
    gpu.context->UpdateSubresource(gpu.constants.Get(), 0, nullptr, &constants, 0, 0);
    ID3D11ShaderResourceView *bound[15]{};
    bound[0] = views[3].Get();
    bound[6] = views[6].Get();
    bound[7] = (offered & candidate::layer) ? views[4].Get() : nullptr;
    bound[11] = (offered & candidate::ui_alpha) ? views[0].Get() : nullptr;
    bound[12] = (offered & candidate::ui_color) ? views[1].Get() : nullptr;
    bound[13] = (offered & candidate::backbuffer) ? views[2].Get() : nullptr;
    bound[14] = (offered & candidate::hudless) ? views[5].Get() : nullptr;
    gpu.context->CSSetShader(tiles, nullptr, 0);
    gpu.context->CSSetShaderResources(0, 15, bound);
    ID3D11Buffer *buffers[3]{nullptr, nullptr, gpu.constants.Get()};
    gpu.context->CSSetConstantBuffers(0, 3, buffers);
    ID3D11UnorderedAccessView *uav = statistics_view.Get();
    gpu.context->CSSetUnorderedAccessViews(6, 1, &uav, nullptr);
    // Each tile in its parts (group z), at columns 16 apart; the CPU adds
    // them as the reduce does.
    gpu.context->Dispatch(16, 16, detection::tile_parts);
    ID3D11UnorderedAccessView *none = nullptr;
    gpu.context->CSSetUnorderedAccessViews(6, 1, &none, nullptr);
    ID3D11ShaderResourceView *cleared[15]{};
    gpu.context->CSSetShaderResources(0, 15, cleared);
    const auto parts = read<texel>(gpu, statistics.Get(), staging.Get(), columns * detection::statistics_rows(0));
    std::vector<texel> rows(16 * detection::statistics_rows(0));
    for (std::size_t row = 0; row != detection::statistics_rows(0); ++row)
      for (std::size_t column = 0; column != 16; ++column)
        for (std::size_t part = 0; part != detection::tile_parts; ++part)
          for (std::size_t k = 0; k != 4; ++k)
            rows[row * 16 + column][k] += parts[row * columns + part * 16 + column][k];
    std::uint32_t total_contradicted = 0;
    texel total_pre_ui{};
    for (std::size_t lane = 0; lane != 256; ++lane) {
      const std::size_t column = lane % 16, row = lane / 16;
      const auto at = [&](const char *what, const texel &got, const texel &want, std::size_t first, std::size_t last) {
        for (std::size_t k = first; k != last; ++k)
          if (got[k] != want[k]) {
            char text[200];
            std::snprintf(text, sizeof(text),
              "tiles pass (%s, flags 0x%x, offered 0x%x, pre-UI threshold %g): tile %zu %s[%zu] is %u, the CPU count %u", space, flags,
              offered, double(pre_ui_threshold), lane, what, k, got[k], want[k]);
            throw std::runtime_error(text);
          }
      };
      at("coverage", rows[row * 16 + column], coverage[lane], 0, 4);
      at("invalid", rows[(row + 16) * 16 + column], invalid[lane], 0, 4);
      at("difference", rows[(row + 32) * 16 + column], difference[lane], 0, 4);
      at("lit and opaque", rows[(row + 48) * 16 + column], lit[lane], 0, 4);
      at("layer and opaque current", rows[(row + detection::layer_statistics_row) * 16 + column], layer[lane], 0, 4);
      at("strong", rows[(row + detection::judgment_statistics_row) * 16 + column], strong_counts[lane], 0, 3);
      at("contradicted", rows[(row + detection::judgment_statistics_row + 16) * 16 + column], contradicted[lane], 0, 3);
      at("pre-UI pixels", rows[(row + detection::pre_ui_statistics_row) * 16 + column], pre_ui[lane], 0, 4);
      for (std::size_t k = 0; k != 3; ++k) total_contradicted += contradicted[lane][k];
      for (std::size_t k = 0; k != 4; ++k) total_pre_ui[k] += pre_ui[lane][k];
    }
    require(!exact || total_contradicted, "check_tiles: the images exercise no contradicted pixel");
    require(pre_ui_threshold > 0.f ? total_pre_ui[0] && total_pre_ui[1] && total_pre_ui[3] && total_pre_ui[0] < 256u * 144u :
      total_pre_ui == texel{}, "check_tiles: the pre-UI pixel counts are degenerate");
  }
}

namespace {
  // S3 semantics, on the CPU (run() proved the reduce equal to decide() on
  // every case): no verdict decides without the gate; exact pairs leave every
  // gated decision unchanged; the gate refuses exactly the non-exact
  // HUD-less pairs, and the reserved bit 0x2 (the removed layer gate)
  // decides nothing.
  void check_identity_semantics(const std::vector<case_t> &cases) {
    const auto same = [](const selection::decision &a, const selection::decision &b) {
      return a.source == b.source && a.covered == b.covered && a.valid_bits == b.valid_bits && a.reused == b.reused &&
        a.refused == b.refused && a.frame_reason == b.frame_reason && a.next.state == b.next.state &&
        a.next.source == b.next.source && a.h1 == b.h1 && a.still == b.still;
    };
    for (const auto &test : cases) {
      const auto c = sums(test.statistics);
      const auto previous = test.previous ? *test.previous : selection::hold_state{};
      const auto plain = selection::decide(c, test.offered, test.accepted, test.flags, previous, test.still);
      auto ungated = test.identity;
      ungated.bits &= identity::token_batch | identity::reserved_gate_layer;
      require(same(plain, selection::decide(c, test.offered, test.accepted, test.flags, previous, test.still, ungated)),
        test.name + ": an identity verdict changed a decision without the gate");
      const selection::identity_input exact{17u, 0u, 16u, 17u, 0u, 17u, identity::gate_hudless};
      require(same(plain, selection::decide(c, test.offered, test.accepted, test.flags, previous, test.still, exact)),
        test.name + ": the gate changed a decision whose pairs are exact");
      const auto gated = selection::decide(c, test.offered, test.accepted, test.flags, previous, test.still, test.identity);
      if ((test.identity.bits & identity::gate_hudless) && !(gated.identity.ok & candidate::hudless))
        require(!(gated.valid_bits & candidate::hudless), test.name + ": a non-exact HUD-less pair left 5/6 valid");
    }
    const auto find = [&](const std::string &name) -> const case_t & {
      for (const auto &test : cases)
        if (test.name == name) return test;
      throw std::runtime_error("S3: no case " + name);
    };
    const auto decided = [](const case_t &test) {
      return selection::decide(sums(test.statistics), test.offered, test.accepted, test.flags,
        test.previous ? *test.previous : selection::hold_state{}, test.still, test.identity);
    };
    const auto late = decided(find("S3 layer pair late copy (mismatch), gate 0"));
    require(late.identity.layer == identity::mismatch && late.identity.layer_delta == 1 &&
        late.identity.layer_space == identity::space_present &&
        decided(find("S3 layer pair far mismatch, gate 0")).identity.layer_delta == 32767 &&
        decided(find("S3 layer pair token mismatch, gate 0")).identity.layer_delta == -1 &&
        decided(find("S3 layer pair unproposed, gate 0")).identity.layer == identity::unproposed,
      "S3: the layer verdict or its delta is wrong");
    const auto hudless_gate = std::to_string(identity::gate_hudless), exact_bit = std::to_string(candidate::exact);
    require(decided(find("S3 HUD-less pair token batch, gate " + hudless_gate + ", exact " + exact_bit)).source == 5u &&
        decided(find("S3 HUD-less pair exact present, gate " + hudless_gate + ", exact " + exact_bit)).source == 5u &&
        decided(find("S3 HUD-less pair generated Present (mismatch), gate 0, exact " + exact_bit)).source == 5u &&
        !decided(find("S3 HUD-less pair generated Present (mismatch), gate " + hudless_gate + ", exact " + exact_bit)).source &&
        !decided(find("S3 HUD-less pair unproposed, gate " + hudless_gate + ", exact 0")).source &&
        decided(find("S3 HUD-less pair unproposed, gate 0, exact 0")).source == 5u,
      "S3: the HUD-less gate does not keep exactly the exact HUD-less pairs");
    const auto full = decided(find("S3 full HUD-less set, mismatch, gate " + hudless_gate));
    require(decided(find("S3 full HUD-less set, mismatch, gate 0")).source == 6u && full.reused && full.source == 5u &&
        full.next.state == detection::hold::spent, "S3: a gated mismatched full set does not take T1's one reuse");
    const auto adds = selection::identity_counter_adds(decided(find("S3 layer pair exact token, gate 0")).identity);
    require(adds[0] == 1u && adds[4] == 1u && adds[1] + adds[2] + adds[3] == 0u,
      "S3: an exact token pair does not count exact and token_exact");
  }
}

int main() {
  try {
    // Pair comparability (V2) from the two snapshots' own encodings.
    const auto threshold = [](std::uint32_t a, std::uint32_t space_a, std::uint32_t b, std::uint32_t space_b) {
      return selection::comparable({a, space_a}, {b, space_b});
    };
    require(threshold(24, 1, 24, 1) == 4.f / 1023.f, "Two R10G10B10A2 snapshots must compare at 4/1023");
    require(threshold(87, 1, 24, 1) == 2.f / 255.f, "Mixed 8/10-bit UNORM must compare at the coarser 2/255");
    require(!threshold(87, 3, 24, 1), "Snapshots of different color spaces must not compare");
    require(threshold(10, 2, 10, 2) == .005f, "Float snapshots under scRGB must compare at .005");
    require(!threshold(10, 2, 24, 2) && !threshold(10, 3, 10, 3), "Float compares only with float under scRGB");
    require(threshold(87, 1, 87, 1) == 2.f / 255.f && threshold(87, 3, 24, 3) == 2.f / 255.f,
      "Same-transfer UNORM pairs must compare at 2/255");
    require(!threshold(29, 1, 28, 1) && threshold(29, 1, 91, 1) == 2.f / 255.f, "sRGB views compare only with sRGB views");
    require(!threshold(0, 1, 24, 1) && !threshold(24, 0, 24, 0), "Unknown formats or color spaces must not compare");
    std::puts("PASS UI pair comparability (V2): same transfer only; 4/1023 for 10/10-bit, 2/255 otherwise, .005 for scRGB float");

    // The acceptance key (A1).
    using selection::signature;
    const signature tag{selection::kind::ui_color, 87, 1};
    require(tag.key() == "ui_color:87:srgb" && signature::parse(tag.key()) == tag, "The tag's signature key does not round-trip");
    require(signature::parse("ui_color:87:pq") != tag && signature::parse(" hudless:24:pq ") == signature{selection::kind::hudless, 24, 3},
      "Signature keys must keep the color space and ignore surrounding blanks");
    for (const char *legacy : {"4", "16", "0x1f", "", "ui_color:87", "ui_color:-1:srgb", "ui_color:87:srgb:x", "layer:10:pq",
           "ui_color:87:sdr"})
      require(!signature::parse(legacy), std::string("A legacy or malformed key parsed: ") + legacy);
    require(std::hash<signature>{}(tag) != std::hash<signature>{}(signature{selection::kind::ui_color, 87, 3}),
      "Signatures of different color spaces must hash apart");
    std::puts("PASS UI acceptance key (A1): kind, DXGI format and color space round-trip; legacy entries never parse");

    // Intended behaviour of the predicate itself (independent of the GPU).
    {
      selection::counts c;
      c.pixels = 1000;
      c.covered = {0, 1000, 5, 1000, 400};
      c.invalid = {0, 0, 0, 0, 0};
      auto d = selection::decide(c, 0x4e, 0x4c, 0);
      require(d.source == 10 && d.covered == 5, "An unaccepted opaque tag blocked the accepted layer");
      d = selection::decide(c, 0x0e, 0x0e, 0);
      require(d.source == 2 && d.covered == 1000, "An accepted full tag did not pin at full coverage (P1)");
      c.invalid[1] = 11;
      d = selection::decide(c, 0x0e, 0x0e, 0);
      require(!d.source && d.none_reason == sunshine_game3d::ui_no_mask::presented_blocked && !d.presented_over_dedicated,
        "An invalid accepted tag did not block accepted presented alpha as presented_blocked");
      d = selection::decide(c, 0x0e, 0x02, 0);
      require(!d.source && d.none_reason == sunshine_game3d::ui_no_mask::trusted_invalid,
        "An invalid accepted tag beside unaccepted presented alpha was not trusted_invalid");
      d = selection::decide(c, 0x0c, 0x00, 0);
      require(!d.source && d.none_reason == sunshine_game3d::ui_no_mask::unaccepted, "Unaccepted selective alpha decided");
      c.covered[0] = 0;
      d = selection::decide(c, 0x09, 0x01, 0);
      require(d.source == 1 && !d.covered, "An accepted empty UIAlpha did not decide empty");
      selection::counts full;
      full.pixels = 1000;
      full.changed = 990;
      full.lit = 900;
      require(selection::decide(full, 0x30, 0x10, 0).source == 6 && !selection::decide(full, 0x30, 0, 0).source &&
          selection::decide(full, 0x30, 0, detection::per_frame_scene_hidden).source == 8 &&
          !selection::decide(full, 0x10, 0x10, 0).source,
        "A full change set must need an accepted exact pair, else H1 under a hidden verdict");
      // An empty change set (revision 8): a lit pair without a changed pixel
      // in clean tiles, exact or not, is valid. Accepted, it decides an empty
      // mask of its own instead of reusing the previous decision (T1).
      selection::counts empty;
      empty.pixels = 1000;
      empty.unchanged = 1000;
      empty.lit = 500;
      empty.matching_tiles = 128;
      const selection::hold_state own{detection::hold::own, 5u, 50u};
      for (const std::uint32_t pair : {0x10u, 0x30u}) {
        const auto accepted = selection::decide(empty, pair, 0x10, 0, own);
        require(accepted.source == 5 && !accepted.covered && !accepted.reused && accepted.own_source == 5 &&
            (accepted.valid_bits & candidate::hudless) && accepted.next.state == detection::hold::own,
          "An accepted empty change set did not decide an empty mask of its own");
        const auto unaccepted = selection::decide(empty, pair, 0x00, 0, own);
        require(!unaccepted.source && !unaccepted.reused && (unaccepted.valid_bits & candidate::hudless),
          "An unaccepted empty change set decided or reused");
      }
      empty.lit = 499;
      require(selection::decide(empty, 0x30, 0x10, 0, own).reused, "A dark empty pair was valid");
      empty.lit = 500;
      empty.matching_tiles = 127;
      require(selection::decide(empty, 0x30, 0x10, 0, own).reused, "A noisy empty pair was valid");
    }
    std::puts("PASS UI selection (S1): accepted valid candidates only, the declared-alpha block, P1 at any coverage");

    // F1: the reason and the refused candidate it names.
    {
      using sunshine_game3d::ui_no_mask::names;
      selection::counts c;
      c.pixels = 1000;
      const auto refusal = [&](std::uint32_t offered, std::uint32_t accepted, std::size_t reason, std::uint32_t refused,
                             const char *what) {
        const auto d = selection::decide(c, offered, accepted, 0);
        require(!d.source && d.none_reason == reason && d.refused == refused && d.frame_reason == reason &&
            selection::frame_reason_name(selection::frame_reason_word(d)) == names[reason],
          std::string("F1 ") + what + ": reason " + std::string(names[d.none_reason]) + " refused 0x" + std::to_string(d.refused));
      };
      c.covered = {0, 0, 300, 300, 300};
      c.invalid = {0, 1000, 0, 0, 0};
      refusal(0x4e, 0x4e, sunshine_game3d::ui_no_mask::presented_blocked, candidate::layer, "presented_blocked");
      c.invalid = {0, 0, 0, 20, 20};
      refusal(0x0c, 0x0c, sunshine_game3d::ui_no_mask::trusted_invalid, candidate::backbuffer, "trusted_invalid");
      c.invalid = {0, 0, 20, 0, 0};
      refusal(0x48, 0x00, sunshine_game3d::ui_no_mask::layer_aside, candidate::layer, "layer_aside");
      c.invalid = {};
      refusal(0x0a, 0x00, sunshine_game3d::ui_no_mask::unaccepted, candidate::current, "unaccepted");
      c.covered = {0, 0, 0, 1000, 0};
      refusal(0x0c, 0x00, sunshine_game3d::ui_no_mask::ambiguous, candidate::backbuffer, "ambiguous");
      refusal(0x10, 0x10, sunshine_game3d::ui_no_mask::difference_failed, candidate::hudless, "difference_failed");
      refusal(0x00, 0x04, sunshine_game3d::ui_no_mask::no_candidate, 0u, "no_candidate");
      c.invalid = {0, 0, 0, 20, 0};
      refusal(0x04, 0x00, sunshine_game3d::ui_no_mask::other, 0u, "other");
      // An acting H1 claim without a held hidden verdict: the first claimant
      // in draw order; an unaccepted opaque UIAlpha never claims.
      c.invalid = {};
      c.covered = {1000, 0, 1000, 0, 0};
      c.opaque_ui_alpha = c.opaque_layer = 1000;
      refusal(0x41, 0x00, sunshine_game3d::ui_no_mask::gate_no_hold, candidate::layer, "layer claim beside UIAlpha");
      refusal(0x40, 0x00, sunshine_game3d::ui_no_mask::gate_no_hold, candidate::layer, "layer claim");
      require(selection::decide(c, 0x01, 0x00, 0).none_reason == sunshine_game3d::ui_no_mask::ambiguous,
        "F1: an unaccepted opaque UIAlpha must not be an H1 claim");
      c.covered = {};
      c.opaque_ui_alpha = c.opaque_layer = 0;
      c.changed = 950;
      // A pre-UI HUD-less claim acts only under the pre-UI hold.
      refusal(0x10, 0x00, sunshine_game3d::ui_no_mask::difference_failed, candidate::hudless, "pre-UI claim without its hold");
      const auto refused = selection::decide(c, 0x10, 0x00, detection::per_frame_pre_ui_visible);
      require(!refused.source && refused.none_reason == sunshine_game3d::ui_no_mask::gate_no_hold &&
          refused.refused == candidate::hudless, "F1: an acting pre-UI claim without the hidden hold must refuse its image");
      const auto decided = selection::decide(c, 0x10, 0x00,
        detection::per_frame_scene_hidden | detection::per_frame_pre_ui_visible);
      require(decided.source == 8 && !decided.refused && decided.frame_reason == detection::frame_reason_decided &&
          selection::frame_reason_name(selection::frame_reason_word(decided)) == "decided",
        "F1: a decided frame must name no refused candidate and the decided frame reason");
    }
    std::puts("PASS UI no-mask reasons (F1): every reason names the first refused candidate in draw order");

    // A2: the one-way judgment needs a valid exact pair and a tenth of the
    // judged source's strong pixels; the counter counts an own decision only.
    {
      selection::counts c;
      c.pixels = 1000;
      c.covered = {0, 0, 0, 1000, 0};
      c.changed = 100;
      c.unchanged = 900;
      c.matching_tiles = 200;
      c.strong = {0, 1000, 0};
      c.contradicted = {0, 100, 0};
      auto d = selection::decide(c, 0x34, 0x04, 0);
      require(d.source == 3 && d.contradicted_bits == candidate::backbuffer && d.contradicted &&
          selection::counter_adds(d, 0)[sunshine_game3d::ui_counter_word::contradicted] == 1,
        "A2: a full Backbuffer over lit unchanged exact-pair pixels was not contradicted");
      c.contradicted[1] = 99;
      require(!selection::decide(c, 0x34, 0x04, 0).contradicted_bits, "A2: fewer than a tenth of strong pixels contradicted");
      c.contradicted[1] = 1000;
      require(!selection::decide(c, 0x14, 0x04, 0).contradicted_bits, "A2: an inexact pair judged");
      c.matching_tiles = 0;
      require(!selection::decide(c, 0x34, 0x04, 0).contradicted_bits, "A2: a middle-band (V2-invalid) pair judged");
      c.matching_tiles = 200;
      d = selection::decide(c, 0x34, 0x00, 0);
      require(d.contradicted_bits == candidate::backbuffer && !d.contradicted,
        "A2: an unaccepted judged source must be contradicted without the decided counter");
      // An exact empty change set (revision 8) judges too.
      c.changed = 0;
      c.unchanged = 1000;
      c.lit = 500;
      require(selection::decide(c, 0x34, 0x04, 0).contradicted_bits == candidate::backbuffer,
        "A2: an exact empty change set did not judge");
    }
    std::puts("PASS UI one-way judgment (A2): an exact valid pair, a tenth of strong pixels lit and unchanged");

    // T1: a real frame without a decision of its own reuses the previous own
    // decision once, then has no mask; a reset never reuses.
    {
      namespace hold = detection::hold;
      namespace counter = sunshine_game3d::ui_counter_word;
      selection::counts c;
      c.pixels = 1000;
      c.covered = {0, 0, 0, 300, 0};
      c.invalid = {0, 0, 0, 20, 0};
      const selection::hold_state own{hold::own, 3u, 1000u};
      auto d = selection::decide(c, 0x04, 0x04, 0, own);
      require(d.reused && d.source == 3 && d.covered == 1000 && !d.own_source && d.full_alpha && d.next.state == hold::spent &&
          d.next.source == 3 && d.none_reason == sunshine_game3d::ui_no_mask::trusted_invalid &&
          selection::frame_reason_word(d) == (sunshine_game3d::ui_no_mask::trusted_invalid | detection::frame_reason_reused),
        "T1: an invalid accepted Backbuffer did not reuse the previous own decision");
      const auto adds = selection::counter_adds(d, 0);
      require(adds[counter::reused] == 1 && adds[counter::decided + 3] == 1 && adds[counter::full_alpha] == 1 &&
          !adds[counter::none + sunshine_game3d::ui_no_mask::trusted_invalid],
        "T1: a reused frame must count the applied decision");
      d = selection::decide(c, 0x04, 0x04, 0, d.next);
      require(!d.reused && !d.source && d.next.state == hold::spent, "T1: the grace was not spent after one reuse");
      require(!selection::decide(c, 0x04, 0x04, detection::per_frame_hold_reset, own).reused, "T1: a reset reused");
      require(!selection::decide(c, 0x04, 0x00, 0, own).reused, "T1: an unaccepted invalid candidate reused");
      const selection::counts empty;
      d = selection::decide(empty, 0, 0x04, detection::per_frame_accepted_missing, own);
      require(d.reused && d.source == 3 && d.none_reason == sunshine_game3d::ui_no_mask::no_candidate,
        "T1: a missing accepted candidate with zero offers did not reuse");
      d = selection::decide(empty, 0, 0x04, 0, own);
      require(!d.reused && d.next.state == hold::own, "T1: a frame offering nothing without accepted_missing has its own decision");
      c.invalid = {};
      d = selection::decide(c, 0x04, 0x04, detection::per_frame_accepted_missing, {hold::spent, 0u, 0u});
      require(d.source == 3 && d.covered == 300 && !d.reused && d.next.state == hold::own,
        "T1: a frame that decides itself must store own whatever is missing");
    }
    std::puts("PASS UI hold grace (T1): one reuse of the previous own decision, then no mask; a reset never reuses");

    // H1: an informative full claim under a held hidden verdict shows the
    // frame flat (8) whatever S1 selected, unless the winner is flat on every
    // pixel already.
    {
      namespace hold = detection::hold;
      const auto hidden = detection::per_frame_scene_hidden, pre_ui = detection::per_frame_pre_ui_visible;
      const auto refuted = [](std::uint32_t bits) { return bits << detection::per_frame_refuted_shift; };
      const auto flat = [](const selection::decision &d) { return d.source == 8u && d.covered == 1000u && d.h1; };
      selection::counts c;
      c.pixels = 1000;
      // (a) An accepted opaque-full UI color tag beside an accepted partial
      // UIAlpha winner (a 0.25% HUD over a full menu): overridden.
      c.covered = {3, 1000, 0, 0, 0};
      c.opaque_ui_color = 990;
      auto d = selection::decide(c, 0x03, 0x03, hidden);
      require(flat(d) && d.s1_source == 1u && d.claims == candidate::ui_color &&
          selection::h1_word(d) == (1u | detection::h1_applied) && d.next.source == 8u,
        "H1 (a): an accepted opaque-full UI color tag did not override the partial UIAlpha winner");
      require(selection::decide(c, 0x03, 0x03, 0).source == 1u, "H1: without a hidden verdict S1 must decide");
      require(selection::decide(c, 0x03, 0x01, hidden).source == 1u, "H1: an unaccepted UI color tag must never claim");
      c.opaque_ui_color = 989;
      require(selection::decide(c, 0x03, 0x03, hidden).source == 1u, "H1: a UI color tag below 99% opaque claimed");
      // (a) only for alpha S1 may select: an accepted opaque-full Backbuffer
      // or current alpha that an accepted declared alpha keeps out of S1
      // (Resident Evil Requiem's presented alpha beside its tag) never claims.
      c = {};
      c.pixels = 1000;
      c.covered = {0, 3, 0, 1000, 1000};
      c.opaque_backbuffer = c.opaque_current = 995;
      d = selection::decide(c, 0x0e, 0x0e, hidden);
      require(d.source == 2u && !d.h1 && !d.claims, "H1 (a): a blocked accepted opaque-full inferred alpha claimed");
      d = selection::decide(c, 0x0c, 0x0c, hidden);
      require(flat(d) && d.claims == (candidate::backbuffer | candidate::current) && d.s1_source == 3u,
        "H1 (a): an accepted opaque-full inferred alpha without a declared one did not claim");
      // An accepted alpha winner opaque on every pixel, and source 6, are
      // flat already; one opaque-full on 99.9% gives way to H1.
      c = {};
      c.pixels = 1000;
      c.covered = {0, 0, 0, 1000, 0};
      c.opaque_backbuffer = 1000;
      d = selection::decide(c, 0x04, 0x04, hidden);
      require(d.source == 3u && !d.h1 && d.claims == candidate::backbuffer, "H1: an accepted winner opaque everywhere was overridden");
      c.opaque_backbuffer = 999;
      d = selection::decide(c, 0x04, 0x04, hidden);
      require(flat(d) && d.s1_source == 3u && d.claims == candidate::backbuffer,
        "H1: an accepted winner with transparent pixels was not shown flat");
      // (b) An unaccepted, valid, opaque-full layer; refuted, or over reused
      // depth, it does not act.
      c = {};
      c.pixels = 1000;
      c.covered = {0, 0, 1000, 0, 1000};
      c.opaque_layer = 990;
      c.opaque_current = 1000;
      require(flat(selection::decide(c, 0x48, 0x00, hidden)), "H1 (b): an unaccepted opaque layer did not flatten");
      d = selection::decide(c, 0x48, 0x00, hidden | refuted(candidate::layer));
      require(!d.source && d.claims == candidate::layer && d.none_reason != sunshine_game3d::ui_no_mask::gate_no_hold,
        "H1: a refuted layer claim acted");
      d = selection::decide(c, 0x48, 0x00, hidden | detection::per_frame_depth_not_current);
      require(!d.source && d.none_reason == sunshine_game3d::ui_no_mask::gate_no_hold && d.refused == candidate::layer,
        "H1: depth that is not this frame's must not apply H1");
      // An unaccepted opaque UIAlpha or UI color tag is never informative.
      c = {};
      c.pixels = 1000;
      c.covered = {1000, 1000, 0, 0, 0};
      c.opaque_ui_alpha = c.opaque_ui_color = 1000;
      d = selection::decide(c, 0x03, 0x00, hidden | pre_ui);
      require(!d.source && !d.claims, "H1: an unaccepted opaque UIAlpha or UI color tag claimed");
      require(selection::decide(c, 0x01, 0x01, hidden).source == 1u, "H1: an accepted opaque UIAlpha must still pin (P1)");
      // (c) An exact full change set, accepted or not.
      c = {};
      c.pixels = 1000;
      c.changed = 990;
      c.lit = 900;
      require(flat(selection::decide(c, 0x30, 0x00, hidden)) && selection::decide(c, 0x30, 0x10, hidden).source == 6u,
        "H1 (c): an unaccepted exact full change set did not flatten, or an accepted one did not stay 6");
      // (d) The pre-UI image: a HUD-less image changed on 90% of pixels or a
      // layer without coverage whose signature is proven (fix 1), acting only
      // under the pre-UI hold.
      c.lit = 0;
      c.changed = 900;
      d = selection::decide(c, 0x10, 0x00, hidden | pre_ui);
      require(flat(d) && d.claims == detection::claim_pre_ui, "H1 (d): a HUD-less pre-UI image did not flatten");
      require(!selection::decide(c, 0x10, 0x00, hidden).source, "H1 (d): a pre-UI claim acted without the pre-UI hold");
      c.changed = 899;
      require(!selection::decide(c, 0x10, 0x00, hidden | pre_ui).source, "H1 (d): a HUD-less image below 90% claimed");
      const auto proven = detection::per_frame_pre_ui_proven;
      c = {};
      c.pixels = 1000;
      c.covered = {0, 0, 0, 0, 1000};
      c.invalid = {0, 0, 600, 0, 0};
      d = selection::decide(c, 0x48, 0x40, hidden | pre_ui | proven);
      require(flat(d) && d.claims == detection::claim_pre_ui, "H1 (d): a proven layer did not flatten under both holds");
      d = selection::decide(c, 0x48, 0x40, hidden | pre_ui);
      require(!d.source && !d.claims && d.none_reason == sunshine_game3d::ui_no_mask::trusted_invalid,
        "H1 (d): an unproven V1-invalid layer claimed (fix 1: V1 validity no longer proves it)");
      d = selection::decide(c, 0x48, 0x40, hidden | proven);
      require(!d.source && d.claims == detection::claim_pre_ui && d.none_reason == sunshine_game3d::ui_no_mask::trusted_invalid,
        "H1 (d): a proven layer without the pre-UI hold must stay as today");
      d = selection::decide(c, 0x48, 0x40, pre_ui | proven);
      require(!d.source && d.none_reason == sunshine_game3d::ui_no_mask::gate_no_hold && d.refused == candidate::layer,
        "F1: a pre-UI layer claim without the hidden hold must refuse the layer");
      c.invalid = {};
      require(selection::decide(c, 0x48, 0x00, hidden | pre_ui | proven).claims == detection::claim_pre_ui,
        "H1 (d): a proven V1-valid layer without coverage is a pre-UI image");
      c.covered[2] = 1;
      d = selection::decide(c, 0x48, 0x40, hidden | pre_ui | proven);
      require(!selection::decide(c, 0x48, 0x00, hidden | pre_ui | proven).claims && d.source == 10u && !d.h1 && !d.claims,
        "H1 (d): a proven layer with coverage claimed");
      // Texel 11 decides nothing.
      c.covered[2] = 0;
      c.pre_ui_match = c.pre_ui_lit = 1000;
      require(flat(selection::decide(c, 0x48, 0x00, hidden | pre_ui | proven)) &&
          !selection::decide(c, 0x48, 0x00, hidden | pre_ui).source, "H1 (d): the pre-UI pixel counts decided");
      // The ledger's proving rule: 90% matching and half lit.
      require(selection::pre_ui_match(900, 500, 1000) && !selection::pre_ui_match(899, 1000, 1000) &&
          !selection::pre_ui_match(1000, 499, 1000) && !selection::pre_ui_match(0, 0, 0),
        "H1 (d): the pre-UI match rule is not 90% matching and half lit");
      // T1 reuses a stored 8 once, like any own decision.
      c = {};
      c.pixels = 1000;
      c.covered = {0, 0, 0, 300, 0};
      c.invalid = {0, 0, 0, 20, 0};
      d = selection::decide(c, 0x04, 0x04, 0, {hold::own, 8u, 1000u});
      require(d.reused && d.source == 8u && !d.h1 && !d.full_alpha, "H1: a stored 8 was not reused by the T1 grace");
    }
    std::puts("PASS UI hidden scene (H1): informative claims (a)-(d) flatten under a held hidden verdict whatever S1 selected; "
      "refuted, reused-depth, unaccepted declared and blocked inferred claims never do");

    // H2 (fix 2): a frame applying no source that T1 did not reuse is flat
    // (11) under still::flatten; S1 (even an empty accepted decision), H1 and
    // T1 win, and the hold store, frame reason and refused candidate keep
    // the decision before it.
    {
      namespace hold = detection::hold;
      namespace counter = sunshine_game3d::ui_counter_word;
      const auto flatten = detection::still::flatten;
      selection::counts c;
      c.pixels = 1000;
      c.covered = {0, 0, 0, 0, 0};
      c.invalid = {0, 0, 600, 0, 0};
      auto d = selection::decide(c, 0x48, 0x00, 0, {}, flatten);
      require(d.source == detection::source_still && d.covered == 1000 && d.still && !d.own_source && !d.h1 &&
          d.next.state == hold::own && d.next.source == 0 && d.next.covered == 0 &&
          d.none_reason == sunshine_game3d::ui_no_mask::layer_aside && d.refused == candidate::layer &&
          selection::frame_reason_word(d) == sunshine_game3d::ui_no_mask::layer_aside,
        "H2: a frame without a UI source was not flat as 11 with its own reason and hold store");
      const auto adds = selection::counter_adds(d, 0);
      require(adds[counter::decided + detection::source_still] == 1 &&
          !adds[counter::none + sunshine_game3d::ui_no_mask::layer_aside] && !adds[counter::full_alpha],
        "H2: counter adds must count decided 11 and no no-mask reason");
      require(!selection::decide(c, 0x48, 0x00, 0, {}, 0).source && !selection::decide(c, 0x48, 0x00, 0, {}, ~flatten).source,
        "H2: decided without its flag");
      require(selection::decide(c, 0x48, 0x00, detection::per_frame_depth_not_current, {}, flatten).source ==
          detection::source_still, "H2: reused depth stopped the still screen");
      selection::counts empty;
      empty.pixels = 1000;
      d = selection::decide(empty, 0x08, 0x08, 0, {}, flatten);
      require(d.source == 4 && !d.covered && !d.still, "H2: an accepted empty decision was overridden");
      c.covered = {0, 0, 1000, 0, 0};
      c.invalid = {};
      c.opaque_layer = 1000;
      d = selection::decide(c, 0x40, 0x00, detection::per_frame_scene_hidden, {}, flatten);
      require(d.source == 8 && d.h1 && !d.still, "H2: H1 was overridden");
      selection::counts invalid;
      invalid.pixels = 1000;
      invalid.covered = {0, 0, 0, 300, 0};
      invalid.invalid = {0, 0, 0, 20, 0};
      d = selection::decide(invalid, 0x04, 0x04, 0, {hold::own, 0u, 0u}, flatten);
      require(d.reused && !d.source && !d.still, "H2: a reused no-mask decision was shown flat");
      d = selection::decide(invalid, 0x04, 0x04, 0, {hold::own, 3u, 400u}, flatten);
      require(d.reused && d.source == 3 && !d.still, "H2: a reused source was overridden");
      d = selection::decide(invalid, 0x04, 0x04, 0, d.next, flatten);
      require(!d.reused && d.source == detection::source_still && d.next.state == hold::spent && d.next.source == 0,
        "H2: a spent grace without a decision was not flat, or stored 11");
    }
    std::puts("PASS UI still screen (H2): a frame applying no source and not reused is flat (11) under the flag; S1, H1 and T1 "
      "win; the hold store, reason and refused candidate keep the decision before it; reused depth does not stop it");

    std::ifstream input(SUNSHINE_GAME3D_NATIVE_HLSL, std::ios::binary);
    require(input.good(), "Cannot read game3d_native.hlsl");
    const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    require(sunshine_game3d::shader_marker(source, selection::revision_marker) == selection::revision &&
        sunshine_game3d::shader_marker(source, detection::candidate_layout_marker) == detection::candidate_layout,
      "game3d_native.hlsl is not the selection revision and candidate layout this test mirrors");
    auto gpu = create_gpu();
    auto cases = crafted();
    const std::size_t crafted_count = cases.size();
    std::mt19937 random(0x5131u);
    for (unsigned i = 0; i != 6000; ++i) cases.push_back(random_case(random, i));
    std::array<unsigned, detection::source_count> sources{};
    unsigned reused = 0, h1 = 0, gate_no_hold = 0, own_retired = 0, still = 0;
    check_stillness(gpu, source);
    std::puts("PASS UI still screen stillness (H2): the compare pass counts compared and still cells (within 1/255 of code luma) "
      "per group, stores each cell's mean, and the evidence pass sums them into texel 12; inactive depth writes nothing");
    // The tiles pass in scRGB (relative tolerances above one), the pre-UI
    // counts at the .005 float pair threshold and without a comparable pair.
    {
      const auto tiles = compile_pass(gpu, source, 2, "SunshineUIDetectionTilesCS");
      for (const float pre_ui_threshold : {.005f, 0.f})
        check_tiles(gpu, tiles.Get(), detection::layer_detection_flags(true), 0x7fu, "scRGB", 2, pre_ui_threshold);
    }
    for (const unsigned color : {1u, 3u}) {
      const auto reduce = compile_pass(gpu, source, color);
      const char *space = color == 1 ? "sRGB" : "PQ";
      const auto tiles = compile_pass(gpu, source, color, "SunshineUIDetectionTilesCS");
      for (const std::uint32_t flags : {0u, detection::layer_detection_flags(false), detection::layer_detection_flags(true)})
        for (const std::uint32_t offered : {0x5fu, 0x7fu})
          for (const float pre_ui_threshold : {2.f / 255.f, 0.f})
            check_tiles(gpu, tiles.Get(), flags, offered, space, color, pre_ui_threshold);
      // Partial offers leave the other candidates unbound, as live renders
      // do: the skipped loads count as the zero an unbound view reads.
      for (const std::uint32_t offered : {0x08u, 0x0cu, 0x49u, 0x3au, 0x16u})
        check_tiles(gpu, tiles.Get(), detection::layer_detection_flags(false), offered, space, color,
          (offered & candidate::layer) ? 2.f / 255.f : 0.f);
      gpu.hold_written = {};
      const std::array<std::uint32_t, detection::hold::store_texels> zero{};
      gpu.context->UpdateSubresource(gpu.hold.Get(), 0, nullptr, zero.data(), UINT(sizeof(zero)), 0);
      for (const auto &test : cases) {
        const auto d = run(gpu, reduce.Get(), test, space);
        if (color == 1) {
          ++sources[d.source < sources.size() ? d.source : 7];
          reused += d.reused;
          h1 += d.h1;
          own_retired += d.own_source == 7u || d.own_source == 9u;
          still += d.still;
          gate_no_hold += !d.own_source && d.none_reason == sunshine_game3d::ui_no_mask::gate_no_hold;
        }
      }
    }
    require(!own_retired && h1 && gate_no_hold && still, "The contract cases decided a retired source (7, 9), or never exercised H1 "
      "or H2");
    check_identity_semantics(cases);
    std::puts("PASS UI identity (S3): the reduce's identity verdicts (texel 12 .z/.w) and identity counter words match "
      "verify_identity on synthetic stamps; without the gate no verdict changes a decision; with exact pairs every gated "
      "decision equals the ungated one; a mismatched HUD-less pair invalidates sources 5 and 6 under gate_hudless (T1 then "
      "reuses once)");
    std::printf("PASS UI detection tiles (V1, V2, A2, H1 d): alpha, layer, difference, one-way and pre-UI pixel counts on edge "
      "values match the CPU count for layer flags 0, 5 and 7, with and without an exact pair and a comparable layer, in sRGB, PQ "
      "and scRGB\n");
    std::printf("PASS UI selection GPU contract (%s): %zu crafted and %zu random cases in two color spaces match decide() in every "
      "decision word, hold store write and counter add; %u reused, %u H1, %u H2, %u gate_no_hold; applied sources",
      gpu.adapter.c_str(), crafted_count, cases.size() - crafted_count, reused, h1, still, gate_no_hold);
    for (std::size_t s = 0; s != sources.size(); ++s)
      if (s != 7 && s != 9) std::printf(" %zu=%u", s, sources[s]);
    std::printf(" other=%u\n", sources[7]);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}

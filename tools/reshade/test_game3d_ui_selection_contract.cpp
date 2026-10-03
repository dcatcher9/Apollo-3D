// SPDX-License-Identifier: GPL-3.0-only
// GPU contract of the UI selection predicate (UI framework S1): runs the real
// SunshineUIDetectionReduceCS of game3d_native.hlsl on synthetic per-tile
// statistics and detection constants, and compares every decision word and
// counter add with ui_selection::decide (game3d_ui_selection.h). The tiles
// pass's alpha and layer counts are checked against a CPU count of V1 on edge
// values. Also checks the predicate's intended behaviour, pair comparability
// (V2) and the acceptance key (A1). Uses a hardware D3D11 device, else WARP.
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
    std::array<texel, 256> coverage{}, invalid{}, difference{}, lit{}, layer{};
  };

  // What the reduce sums from the tiles, exactly as it does.
  selection::counts sums(const tiles &t) {
    selection::counts c;
    std::array<std::uint32_t, 4> coverage{}, invalid{}, difference{};
    std::array<std::uint32_t, 3> lit{}, layer{};
    for (std::size_t i = 0; i != 256; ++i) {
      for (std::size_t k = 0; k != 4; ++k) {
        coverage[k] += t.coverage[i][k];
        invalid[k] += t.invalid[i][k];
        difference[k] += t.difference[i][k];
      }
      for (std::size_t k = 0; k != 3; ++k) {
        lit[k] += t.lit[i][k];
        layer[k] += t.layer[i][k];
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
    return c;
  }

  struct case_t {
    std::string name;
    tiles statistics;
    std::uint32_t offered{}, accepted{}, flags{};
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
    ComPtr<ID3D11Texture2D> statistics, decision, counters, decision_staging, counters_staging;
    ComPtr<ID3D11ShaderResourceView> statistics_view;
    ComPtr<ID3D11UnorderedAccessView> decision_view, counters_view;
    ComPtr<ID3D11Buffer> constants;
    std::string adapter;
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
    desc.Width = 16;
    desc.Height = detection::statistics_rows(0);
    desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.statistics), "statistics");
    checked(gpu.device->CreateShaderResourceView(gpu.statistics.Get(), nullptr, &gpu.statistics_view), "statistics view");
    desc.Width = detection::layer_decision_texels;
    desc.Height = 1;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.decision), "decision");
    checked(gpu.device->CreateUnorderedAccessView(gpu.decision.Get(), nullptr, &gpu.decision_view), "decision view");
    desc.BindFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    checked(gpu.device->CreateTexture2D(&desc, nullptr, &gpu.decision_staging), "decision staging");
    desc.Width = UINT(count);
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
    D3D11_BUFFER_DESC buffer{};
    buffer.ByteWidth = 16;
    buffer.Usage = D3D11_USAGE_DEFAULT;
    buffer.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    checked(gpu.device->CreateBuffer(&buffer, nullptr, &gpu.constants), "constants");
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

  // Runs one case and compares it; returns the decision for the semantic checks.
  selection::decision run(gpu_t &gpu, ID3D11ComputeShader *reduce, const case_t &test, const char *space) {
    std::array<texel, 16 * 80> rows{};
    for (std::size_t lane = 0; lane != 256; ++lane) {
      const std::size_t column = lane % 16, row = lane / 16;
      rows[row * 16 + column] = test.statistics.coverage[lane];
      rows[(row + 16) * 16 + column] = test.statistics.invalid[lane];
      rows[(row + 32) * 16 + column] = test.statistics.difference[lane];
      rows[(row + 48) * 16 + column] = test.statistics.lit[lane];
      rows[(row + detection::layer_statistics_row) * 16 + column] = test.statistics.layer[lane];
    }
    gpu.context->UpdateSubresource(gpu.statistics.Get(), 0, nullptr, rows.data(), 16 * sizeof(texel), 0);
    struct {
      std::uint32_t offered;
      float threshold;
      std::uint32_t accepted, flags;
    } constants{test.offered, 2.f / 255.f, test.accepted, test.flags};
    gpu.context->UpdateSubresource(gpu.constants.Get(), 0, nullptr, &constants, 0, 0);
    const std::array<std::uint32_t, 4> sentinel{0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
    gpu.context->ClearUnorderedAccessViewUint(gpu.decision_view.Get(), sentinel.data());
    const std::array<std::uint32_t, 4> zero{};
    gpu.context->ClearUnorderedAccessViewUint(gpu.counters_view.Get(), zero.data());
    gpu.context->CSSetShader(reduce, nullptr, 0);
    ID3D11ShaderResourceView *views[11]{};
    views[10] = gpu.statistics_view.Get();
    gpu.context->CSSetShaderResources(0, 11, views);
    ID3D11Buffer *buffers[3]{nullptr, nullptr, gpu.constants.Get()};
    gpu.context->CSSetConstantBuffers(0, 3, buffers);
    ID3D11UnorderedAccessView *uavs[2]{gpu.decision_view.Get(), gpu.counters_view.Get()};
    gpu.context->CSSetUnorderedAccessViews(6, 2, uavs, nullptr);
    gpu.context->Dispatch(1, 1, 1);
    ID3D11UnorderedAccessView *none[2]{};
    gpu.context->CSSetUnorderedAccessViews(6, 2, none, nullptr);
    const auto words = read<std::uint32_t>(gpu, gpu.decision.Get(), gpu.decision_staging.Get(), 4 * detection::layer_decision_texels);
    const auto counters = read<std::uint32_t>(gpu, gpu.counters.Get(), gpu.counters_staging.Get(), count);

    const auto c = sums(test.statistics);
    const auto d = selection::decide(c, test.offered, test.accepted, test.flags);
    std::array<std::uint32_t, 4 * detection::layer_decision_texels> want{};
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
    // decide() reads its counts back from the words it is compared with.
    const auto back = selection::counts_from_words(want.data(), want.size());
    require(back.covered == c.covered && back.invalid == c.invalid && back.pixels == c.pixels &&
        back.matching_tiles == c.matching_tiles && back.opaque_layer == c.opaque_layer,
      test.name + ": counts_from_words does not invert the decision words");
    for (std::size_t i = 0; i != want.size(); ++i)
      if (words[i] != want[i]) {
        char text[200];
        std::snprintf(text, sizeof(text), " (%s): word %zu is %u on the GPU, decide() gives %u (offered 0x%x accepted 0x%x flags 0x%x)",
          space, i, words[i], want[i], test.offered, test.accepted, test.flags);
        throw std::runtime_error(test.name + text);
      }
    const auto adds = selection::counter_adds(d, test.flags);
    for (std::size_t i = 0; i != count; ++i)
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
      case selection::kind::backbuffer: t.coverage[0][2] = covered; t.invalid[0][2] = invalid; break;
      case selection::kind::current: t.coverage[0][3] = covered; t.invalid[0][3] = invalid; break;
      case selection::kind::ui_layer: t.layer[0] = {covered, invalid, opaque, 0}; break;
      default: break;
    }
  }

  std::vector<case_t> crafted() {
    std::vector<case_t> cases;
    const std::uint32_t p = small_pixels;
    using k = selection::kind;
    const auto hold = detection::per_frame_scene_hold, hold_hudless = detection::per_frame_scene_hold_hudless;
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
    // and without the HUD-less hold, and inexact.
    for (const std::uint32_t permille : {979u, 980u, 1000u})
      for (const std::uint32_t accepted : {0u, unsigned(candidate::hudless)})
        for (const std::uint32_t flags : {0u, hold_hudless})
          for (const std::uint32_t exact : {0u, unsigned(candidate::exact)}) {
            case_t test{"full change set " + std::to_string(permille), {}, candidate::hudless | exact, accepted, flags};
            difference_rows(test.statistics, p, 0, permille, 0);
            test.statistics.lit[0][0] = p / 2u;
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
    // Every route flag over each gate input, accepted or not.
    for (const std::uint32_t offered : {unsigned(candidate::ui_alpha), unsigned(candidate::layer),
           unsigned(candidate::ui_alpha | candidate::layer), unsigned(candidate::ui_color)})
      for (const std::uint32_t accepted : {0u, offered})
        for (const std::uint32_t flags : {0u, hold, hold_hudless, hold | hold_hudless})
          for (const std::uint32_t invalid : {0u, 1u}) {
            auto test = alpha_case("route gate", p, offered | candidate::hudless, accepted, flags);
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
    static constexpr std::array<std::uint32_t, 3> per_frame{detection::per_frame_scene_hold,
      detection::per_frame_scene_hold_hudless, detection::per_frame_depth_not_current};
    for (const auto bit : per_frame)
      if (random() & 1u) test.flags |= bit;
    test.flags |= random() & 7u;
    auto &t = test.statistics;
    for (std::size_t k = 0; k != 4; ++k) {
      spread(t.coverage, k, pick(random, p), random);
      spread(t.invalid, k, pick(random, p) / ((random() & 3u) ? 64u : 1u), random);
    }
    for (std::size_t k = 0; k != 3; ++k) spread(t.layer, k, k == 1 ? pick(random, p) / ((random() & 3u) ? 64u : 1u) :
      pick(random, p), random);
    spread(t.lit, 0, pick(random, p), random);
    spread(t.lit, 1, pick(random, p), random);
    spread(t.lit, 2, pick(random, p), random);
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
    return test;
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

  // The tiles pass's alpha and layer statistics (rows 0-31, the opaque
  // components of rows 48-63, and the layer's rows 64-79) on images of edge
  // values, against a CPU count of the V1 rules: covered is in (0, 1],
  // invalid is non-finite or out of range, opaque at least 254/255, and the
  // layer alone also fails beyond the premultiplied bound of its flags.
  void check_tiles(gpu_t &gpu, ID3D11ComputeShader *tiles, std::uint32_t flags, const char *space) {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const std::array<float, 12> alphas{0.f, -0.f, 1e-30f, .5f, 1.f, 1.5f, -.5f, nan, inf, .999f, .99f, .25f};
    const std::array<std::array<float, 4>, 9> layers{{{0.f, 0.f, 0.f, 0.f}, {.5f, .5f, .5f, .5f}, {1.f, 1.f, 1.f, .1f},
      {.3f, 0.f, 0.f, 0.f}, {nan, 0.f, 0.f, .2f}, {0.f, 0.f, 0.f, -0.f}, {20.f, 0.f, 0.f, .5f}, {.2f, .2f, .2f, .999f},
      {0.f, 0.f, 0.f, nan}}};
    std::array<std::vector<std::array<float, 4>>, 5> images; // t11, t12, t13, t0, t7
    for (auto &image : images) image.resize(256 * 144);
    for (std::uint32_t y = 0; y != 144; ++y)
      for (std::uint32_t x = 0; x != 256; ++x) {
        const auto i = y * 256 + x;
        images[0][i] = {alphas[(x + y) % alphas.size()], 0.f, 0.f, 1.f};
        images[1][i] = {0.f, 0.f, 0.f, alphas[(x * 3 + y) % alphas.size()]};
        images[2][i] = {0.f, 0.f, 0.f, alphas[(x + 5 * y) % alphas.size()]};
        images[3][i] = {.5f, .5f, .5f, alphas[(x * 7 + y * 2) % alphas.size()]};
        images[4][i] = layers[(x + 2 * y) % layers.size()];
      }
    const auto okay = [](float a) { return std::isfinite(a) && a >= 0.f && a <= 1.f; };
    const float opaque = 254.f / 255.f, headroom = (flags & detection::stored_hdr_headroom) ? 125.f : 2.f;
    std::array<texel, 256> coverage{}, invalid{}, lit{}, layer{};
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
        const auto &l = images[4][i];
        const bool bound = (flags & detection::stored_premultiplied) && std::max({l[0], l[1], l[2]}) > l[3] * headroom + 4.f / 255.f;
        layer[tile][0] += okay(l[3]) && l[3] > 0.f;
        layer[tile][1] += !okay(l[3]) || bound;
        layer[tile][2] += okay(l[3]) && l[3] >= opaque;
      }
    std::array<ComPtr<ID3D11ShaderResourceView>, 5> views;
    for (std::size_t k = 0; k != views.size(); ++k) views[k] = float_image(gpu, images[k]);
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = 16;
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
    struct {
      std::uint32_t offered;
      float threshold;
      std::uint32_t accepted, flags;
    } constants{0x4fu, 2.f / 255.f, 0u, flags};
    gpu.context->UpdateSubresource(gpu.constants.Get(), 0, nullptr, &constants, 0, 0);
    ID3D11ShaderResourceView *bound[15]{};
    bound[0] = views[3].Get();
    bound[7] = views[4].Get();
    bound[11] = views[0].Get();
    bound[12] = views[1].Get();
    bound[13] = views[2].Get();
    gpu.context->CSSetShader(tiles, nullptr, 0);
    gpu.context->CSSetShaderResources(0, 15, bound);
    ID3D11Buffer *buffers[3]{nullptr, nullptr, gpu.constants.Get()};
    gpu.context->CSSetConstantBuffers(0, 3, buffers);
    ID3D11UnorderedAccessView *uav = statistics_view.Get();
    gpu.context->CSSetUnorderedAccessViews(6, 1, &uav, nullptr);
    gpu.context->Dispatch(16, 16, 1);
    ID3D11UnorderedAccessView *none = nullptr;
    gpu.context->CSSetUnorderedAccessViews(6, 1, &none, nullptr);
    ID3D11ShaderResourceView *cleared[15]{};
    gpu.context->CSSetShaderResources(0, 15, cleared);
    const auto rows = read<texel>(gpu, statistics.Get(), staging.Get(), 16 * detection::statistics_rows(0));
    for (std::size_t lane = 0; lane != 256; ++lane) {
      const std::size_t column = lane % 16, row = lane / 16;
      const auto &got_lit = rows[(row + 48) * 16 + column];
      const auto at = [&](const char *what, const texel &got, const texel &want, std::size_t first, std::size_t last) {
        for (std::size_t k = first; k != last; ++k)
          if (got[k] != want[k]) {
            char text[160];
            std::snprintf(text, sizeof(text), "tiles pass (%s, flags 0x%x): tile %zu %s[%zu] is %u, the V1 count %u", space, flags,
              lane, what, k, got[k], want[k]);
            throw std::runtime_error(text);
          }
      };
      at("coverage", rows[row * 16 + column], coverage[lane], 0, 4);
      at("invalid", rows[(row + 16) * 16 + column], invalid[lane], 0, 4);
      at("opaque", got_lit, lit[lane], 1, 3);
      at("layer", rows[(row + detection::layer_statistics_row) * 16 + column], layer[lane], 0, 3);
    }
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
          selection::decide(full, 0x30, 0, detection::per_frame_scene_hold_hudless).source == 9 &&
          !selection::decide(full, 0x10, 0x10, 0).source,
        "A full change set must need an accepted exact pair, else the held HUD-less route");
    }
    std::puts("PASS UI selection (S1): accepted valid candidates only, the declared-alpha block, P1 at any coverage");

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
    for (const unsigned color : {1u, 3u}) {
      const auto reduce = compile_pass(gpu, source, color);
      const char *space = color == 1 ? "sRGB" : "PQ";
      const auto tiles = compile_pass(gpu, source, color, "SunshineUIDetectionTilesCS");
      for (const std::uint32_t flags : {0u, detection::layer_detection_flags(false), detection::layer_detection_flags(true)})
        check_tiles(gpu, tiles.Get(), flags, space);
      for (const auto &test : cases) {
        const auto d = run(gpu, reduce.Get(), test, space);
        if (color == 1) ++sources[d.source];
      }
    }
    std::printf("PASS UI detection tiles (V1): alpha and layer counts on edge values match the CPU count for layer flags 0, 5 "
      "and 7 in two color spaces\n");
    std::printf("PASS UI selection GPU contract (%s): %zu crafted and %zu random cases in two color spaces match decide() in every "
      "decision word and counter add; sources", gpu.adapter.c_str(), crafted_count, cases.size() - crafted_count);
    for (std::size_t s = 0; s != sources.size(); ++s)
      if (s != 7) std::printf(" %zu=%u", s, sources[s]);
    std::printf("\n");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL %s\n", error.what());
    return 1;
  }
}

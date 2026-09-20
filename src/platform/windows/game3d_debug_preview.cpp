// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_debug_preview.h"
#include "src/game3d_debug_protocol.h"
#include "src/game3d_debug_ui_resources.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <wincodec.h>
#include <windows.h>
#include <wrl/client.h>

namespace platf::game3d_debug::preview {
  namespace {
    using json = nlohmann::json;
    using detail::rgb;
    using Microsoft::WRL::ComPtr;
    constexpr rgb invalid_color {255, 0, 255};
    constexpr std::uint64_t max_bytes = ::game3d_debug::max_capture_bytes;

    void check(HRESULT result, const char *operation) {
      if (FAILED(result)) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(static_cast<unsigned long>(result)));
      }
    }

    struct com_scope {
      HRESULT result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);

      ~com_scope() {
        if (SUCCEEDED(result)) {
          CoUninitialize();
        }
      }
    };

    void png(IWICImagingFactory *factory, const std::filesystem::path &path, unsigned width, unsigned height, const std::function<rgb(unsigned, unsigned)> &pixel) {
      ComPtr<IWICStream> stream;
      check(factory->CreateStream(&stream), "Create PNG stream");
      check(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE), "Open PNG");
      ComPtr<IWICBitmapEncoder> encoder;
      check(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder), "Create PNG encoder");
      check(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache), "Initialize PNG encoder");
      ComPtr<IWICBitmapFrameEncode> frame;
      ComPtr<IPropertyBag2> properties;
      check(encoder->CreateNewFrame(&frame, &properties), "Create PNG frame");
      check(frame->Initialize(properties.Get()), "Initialize PNG frame");
      check(frame->SetSize(width, height), "Set PNG dimensions");
      WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
      check(frame->SetPixelFormat(&format), "Set PNG format");
      if (!IsEqualGUID(format, GUID_WICPixelFormat24bppBGR)) {
        throw std::runtime_error("PNG encoder changed pixel format");
      }
      std::vector<std::uint8_t> row(width * 3);
      for (unsigned y = 0; y < height; ++y) {
        for (unsigned x = 0; x < width; ++x) {
          const auto value = pixel(x, y);
          row[x * 3] = value[2];
          row[x * 3 + 1] = value[1];
          row[x * 3 + 2] = value[0];
        }
        check(frame->WritePixels(1, static_cast<UINT>(row.size()), static_cast<UINT>(row.size()), row.data()), "Write PNG row");
      }
      check(frame->Commit(), "Commit PNG frame");
      check(encoder->Commit(), "Commit PNG");
    }

    struct range {
      float low = std::numeric_limits<float>::infinity();
      float high = -std::numeric_limits<float>::infinity();
      std::uint64_t finite = 0, nonfinite = 0;

      void add(float value) {
        if (!std::isfinite(value)) {
          ++nonfinite;
          return;
        }
        low = std::min(low, value);
        high = std::max(high, value);
        ++finite;
      }

      json describe() const {
        return {{"finite_pixels", finite}, {"nonfinite_pixels", nonfinite}, {"minimum", finite ? json(low) : json(nullptr)}, {"maximum", finite ? json(high) : json(nullptr)}, {"nonfinite_color", "magenta"}};
      }

      float signed_limit() const {
        return finite ? std::max(std::abs(low), std::abs(high)) : 0;
      }
    };

    range scalar_range(const image_view &image, unsigned channel = 0) {
      range result;
      for (unsigned y = 0; y < image.height; ++y) {
        for (unsigned x = 0; x < image.width; ++x) {
          result.add(detail::sample(image, x, y)[channel]);
        }
      }
      return result;
    }

    const sunshine_game3d::ui_resources::descriptor *optional_resource(const std::string &kind) {
      for (const auto &entry : sunshine_game3d::ui_resources::catalog)
        if (kind == entry.file_stem) return &entry;
      return nullptr;
    }

    bool primary_resource(const std::string &kind) {
      for (const auto *name : {"source_color", "raw_depth", "candidate", "vertical_majorant",
             "vertical_field", "final_field", "sbs", "linear_color", "ui_source_color"})
        if (kind == name) return true;
      return false;
    }

    unsigned channel_count(unsigned format) {
      switch (format) {
        case 2: case 10: case 24: case 28: case 29: case 87: case 91: return 4;
        case 16: case 34: return 2;
        default: return 1;
      }
    }

    bool real_alpha_channel(unsigned format) {
      // Only decoded formats with an actual A component; never a decoder's
      // synthetic default or the padding byte of an RGBX resource.
      switch (format) {
        case 2: case 10: case 24: case 28: case 29: case 87: case 91: return true;
        default: return false;
      }
    }

    json captured_resource(const json &metadata, unsigned id) {
      const auto it = metadata.find("optional_captures");
      if (it != metadata.end() && it->is_array())
        for (const auto &entry : *it)
          if (entry.is_object() && entry.contains("artifact_id") && entry["artifact_id"] == id) return entry;
      return json::object();
    }

    std::string escape(const std::string &text) {
      std::string result;
      for (const char ch : text) {
        switch (ch) {
          case '&':
            result += "&amp;";
            break;
          case '<':
            result += "&lt;";
            break;
          case '>':
            result += "&gt;";
            break;
          case '"':
            result += "&quot;";
            break;
          default:
            result += ch;
        }
      }
      return result;
    }

    float half(std::uint16_t bits) {
      const float magnitude = (bits & 0x7c00) == 0      ? std::ldexp(static_cast<float>(bits & 1023), -24) :
                              (bits & 0x7c00) == 0x7c00 ? ((bits & 1023) ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity()) :
                                                          std::ldexp(1.0f + static_cast<float>(bits & 1023) / 1024, static_cast<int>((bits >> 10) & 31) - 15);
      return bits & 0x8000 ? -magnitude : magnitude;
    }

    template<class T>
    T load(const std::uint8_t *p) {
      T value;
      std::memcpy(&value, p, sizeof(value));
      return value;
    }

    std::uint8_t byte(float value) {
      return static_cast<std::uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255));
    }
  }  // namespace

  namespace detail {
    std::uint32_t bytes_per_pixel(std::uint32_t format) {
      switch (format) {
        case 2:
          return 16;
        case 10:
        case 16:
          return 8;
        case 24:
        case 28:
        case 29:
        case 34:
        case 41:
        case 87:
        case 91:
          return 4;
        case 54:
        case 56:
          return 2;
        case 61:
        case 62:
          return 1;
        default:
          return 0;
      }
    }

    std::array<float, 4> sample(const image_view &image, unsigned x, unsigned y) {
      const auto *p = image.bytes.data() + (static_cast<std::size_t>(y) * image.width + x) * bytes_per_pixel(image.dxgi_format);
      std::array<float, 4> out {0, 0, 0, 1};
      switch (image.dxgi_format) {
        case 2:
        case 16:
        case 41:
          {
            const unsigned channels = image.dxgi_format == 2 ? 4 : image.dxgi_format == 16 ? 2 :
                                                                                             1;
            for (unsigned c = 0; c < channels; ++c) {
              out[c] = load<float>(p + c * 4);
            }
            break;
          }
        case 10:
        case 34:
        case 54:
          {
            const unsigned channels = image.dxgi_format == 10 ? 4 : image.dxgi_format == 34 ? 2 :
                                                                                              1;
            for (unsigned c = 0; c < channels; ++c) {
              out[c] = half(load<std::uint16_t>(p + c * 2));
            }
            break;
          }
        case 24:
          {
            const auto packed = load<std::uint32_t>(p);
            for (unsigned c = 0; c < 3; ++c) {
              out[c] = ((packed >> (c * 10)) & 1023) / 1023.0f;
            }
            out[3] = (packed >> 30) / 3.0f;
            break;
          }
        case 28:
        case 29:
        case 87:
        case 91:
          for (unsigned c = 0; c < 4; ++c) {
            out[c] = p[c] / 255.0f;
          }
          if (image.dxgi_format == 87 || image.dxgi_format == 91) {
            std::swap(out[0], out[2]);
          }
          break;
        case 56:
          out[0] = load<std::uint16_t>(p) / 65535.0f;
          break;
        case 61:
          out[0] = p[0] / 255.0f;
          break;
        case 62:
          out[0] = p[0]; // UINT is a literal integer, not normalized coverage.
          break;
        default:
          throw std::runtime_error("Unsupported preview format");
      }
      return out;
    }

    rgb display_color(std::array<float, 4> value, unsigned color_space) {
      if (!std::isfinite(value[0]) || !std::isfinite(value[1]) || !std::isfinite(value[2])) {
        return invalid_color;
      }
      if (color_space == 3) {
        for (unsigned c = 0; c < 3; ++c) {
          const float power = std::pow(std::clamp(value[c], 0.0f, 1.0f), 1.0f / 78.84375f);
          value[c] = 125 * std::pow(std::max(power - 0.8359375f, 0.0f) / std::max(18.8515625f - 18.6875f * power, 1e-8f), 1.0f / 0.1593017578125f);
        }
        const auto input = value;
        value[0] = 1.6604910021f * input[0] - 0.5876411388f * input[1] - 0.0728498633f * input[2];
        value[1] = -0.1245504745f * input[0] + 1.1328998971f * input[1] - 0.0083494226f * input[2];
        value[2] = -0.0181507634f * input[0] - 0.1005788980f * input[1] + 1.1187296614f * input[2];
      }
      if (color_space == 2 || color_space == 3) {
        for (unsigned c = 0; c < 3; ++c) {
          const float linear = std::max(value[c], 0.0f) / (1.0f + std::max(value[c], 0.0f));
          value[c] = linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
        }
      }
      return {byte(value[0]), byte(value[1]), byte(value[2])};
    }

    rgb scalar_color(float value, float low, float high) {
      if (!std::isfinite(value)) {
        return invalid_color;
      }
      const auto shade = high > low ? byte((value - low) / (high - low)) : std::uint8_t(128);
      return {shade, shade, shade};
    }

    rgb signed_color(float value, float limit) {
      if (!std::isfinite(value)) {
        return invalid_color;
      }
      const float t = limit > 0 ? std::clamp(std::abs(value) / limit, 0.0f, 1.0f) : 0;
      constexpr rgb negative {45, 110, 245}, positive {240, 95, 35}, neutral {225, 225, 225};
      const auto &end = value < 0 ? negative : positive;
      rgb result;
      for (unsigned c = 0; c < 3; ++c) {
        result[c] = static_cast<std::uint8_t>(std::lround(neutral[c] + t * (float(end[c]) - neutral[c])));
      }
      return result;
    }

    std::array<unsigned, 2> aligned_depth_pixel(unsigned x, unsigned y, unsigned color_width, unsigned color_height, unsigned depth_width, unsigned depth_height, const std::array<float, 4> &rect, const std::array<float, 2> &jitter) {
      std::array<unsigned, 2> out;
      for (unsigned axis = 0; axis < 2; ++axis) {
        const unsigned input = axis ? y : x, size = axis ? depth_height : depth_width, color_size = axis ? color_height : color_width;
        const float first = rect[axis] + 0.5f / size;
        const float last = rect[axis] + rect[axis + 2] - 0.5f / size;
        const float uv = std::clamp(rect[axis] + ((input + 0.5f) / color_size) * rect[axis + 2] + jitter[axis], first, std::max(first, last));
        out[axis] = static_cast<unsigned>(std::clamp(std::floor(uv * size), 0.0f, float(size - 1)));
      }
      return out;
    }

    bool safe_artifact_filename(const std::string &name) {
      // Native artifacts use .bin. This also excludes every generated preview,
      // HTML and metadata name so backfilling can never overwrite an input.
      const bool binary_extension = name.size() > 4 && name[name.size() - 4] == '.' &&
                                    (name[name.size() - 3] == 'b' || name[name.size() - 3] == 'B') &&
                                    (name[name.size() - 2] == 'i' || name[name.size() - 2] == 'I') &&
                                    (name[name.size() - 1] == 'n' || name[name.size() - 1] == 'N');
      return binary_extension && std::all_of(name.begin(), name.end(), [](unsigned char c) {
               return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
             });
    }
  }  // namespace detail

  json generate(const std::filesystem::path &directory, const json &manifest, std::span<const image_view> inputs) {
    std::uint64_t total = 0;
    if (inputs.size() > ::game3d_debug::max_textures) {
      throw std::runtime_error("Too many preview inputs");
    }
    std::vector<image_view> accepted;
    json optional_errors = json::array();
    for (bool optional : {false, true}) {
      for (const auto &image : inputs) {
        if ((!primary_resource(image.kind)) != optional) continue;
        const auto reject = [&](const char *reason) {
          if (!optional) throw std::runtime_error(reason);
          optional_errors.push_back({{"kind", image.kind}, {"reason", reason}});
        };
        if (optional && !optional_resource(image.kind)) {
          reject("Unknown optional resource kind");
          continue;
        }
        const auto size = std::uint64_t(image.width) * image.height * detail::bytes_per_pixel(image.dxgi_format);
        if (!image.width || !image.height || image.width > 16384 || image.height > 16384 || !size || size != image.bytes.size() || size > max_bytes || total > max_bytes - size) {
          reject("Invalid preview input dimensions, packing or byte budget");
          continue;
        }
        if (std::any_of(accepted.begin(), accepted.end(), [&](const image_view &other) { return other.kind == image.kind; })) {
          reject("Duplicate preview input kind");
          continue;
        }
        total += size;
        accepted.push_back(image);
      }
    }
    const std::span<const image_view> images(accepted);
    com_scope com;
    if (FAILED(com.result) && com.result != RPC_E_CHANGED_MODE) {
      check(com.result, "Initialize preview COM");
    }
    ComPtr<IWICImagingFactory> factory;
    check(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory)), "Create preview WIC factory");
    std::filesystem::create_directories(directory);
    const auto metadata = manifest.value("producer_metadata", json::object());
    const auto parameters = metadata.value("render_parameters", json::object());
    const unsigned color_space = metadata.value("color_space", 0u);
    json report = {{"schema", "sunshine.game3d.previews.v1"}, {"status", "complete"}, {"index", "index.html"}, {"images", json::array()}, {"missing", json::array()}, {"notes", "Display-only PNGs. Native .bin and manifest replay parameters are authoritative and unchanged. Magenta marks nonfinite values. No percentile clipping."}};
    report["optional_errors"] = std::move(optional_errors);
    report["ui_resources"] = metadata.value("ui_resources", json::array());
    report["optional_captures"] = metadata.value("optional_captures", json::array());
    auto find = [&](const char *kind) -> const image_view * {
      for (const auto &image : images) {
        if (image.kind == kind) {
          return &image;
        }
      }
      return nullptr;
    };
    auto add = [&](const char *file, const std::string &title, unsigned width, unsigned height, const std::string &caption, const std::function<rgb(unsigned, unsigned)> &pixel, json extra = json::object()) {
      png(factory.Get(), directory / file, width, height, pixel);
      extra.update({{"file", file}, {"title", title}, {"width", width}, {"height", height}, {"caption", caption}});
      report["images"].push_back(std::move(extra));
    };
    const auto *color = find("source_color"), *depth = find("raw_depth"), *sbs = find("sbs"), *field = find("final_field"), *candidate = find("candidate");
    for (const auto *required : {"source_color", "raw_depth", "sbs"}) {
      if (!find(required)) {
        report["missing"].push_back(required);
      }
    }
    const std::string hdr_note = "Display-only HDR mapping: PQ sources decode ST.2084 and convert Rec.2020 to Rec.709; scRGB uses 1 = 80 nits. Negative RGB is clipped for display; positive RGB uses c/(1+c), then sRGB encoding. This is not the host display transform.";
    const std::string encoded_note = color_space == 1 ? "Encoded SDR sRGB values preserved, rounded only to PNG's 8-bit RGB." : "Unknown transfer function: RGB codes displayed unchanged, rounded to 8-bit. This is not a verified SDR/HDR appearance.";
    if (color) {
      add("color.png", "Raw color before cropping", color->width, color->height, (color_space == 2 || color_space == 3 ? hdr_note : encoded_note) + " Full source allocation; no crop or jitter applied. Alpha is ignored for display.", [&](unsigned x, unsigned y) {
        return detail::display_color(detail::sample(*color, x, y), color_space);
      });
      if (real_alpha_channel(color->dxgi_format)) {
        add("source_alpha.png", "Raw source color alpha", color->width, color->height,
          "A channel of the original source_color allocation: 0 black, 1 white; constant values remain visible and nonfinite values are magenta. No color transfer, crop, jitter or threshold is applied. UI meaning is not guaranteed; this may be UI coverage, constant data or another game-defined value. This is not SBS output alpha. Generating this preview does not enable source-alpha UI protection.",
          [&](unsigned x, unsigned y) { return detail::scalar_color(detail::sample(*color, x, y)[3], 0, 1); },
          {{"source_artifact", "source_color"}, {"channel", "A"}, {"black", 0}, {"white", 1},
            {"semantic", "uninterpreted_source_alpha"}, {"range", scalar_range(*color, 3).describe()}});
      }
    }
    if (const auto *ui = find("ui_source_color")) {
      add("ui_source_color.png", "Consumed UI-alpha source (RGB diagnostic only)", ui->width, ui->height,
        "Exact RGBA input consumed for horizontal-pass alpha. RGB code values shown without assuming its transfer function; it never replaces the current eye color. See ui_source provenance for timing and selection.",
        [&](unsigned x, unsigned y) { return detail::display_color(detail::sample(*ui, x, y), 0); },
        {{"source_artifact", "ui_source_color"}, {"provenance", metadata.value("ui_source", json::object())}});
      if (real_alpha_channel(ui->dxgi_format)) {
        add("ui_source_alpha.png", "Consumed pre-FG UI alpha", ui->width, ui->height,
          "Exact consumed RGBA alpha: fixed 0 black, 1 white, nonfinite magenta. No transfer, crop, jitter or threshold. This is not the latest optional SL Backbuffer snapshot and does not prove same-game-frame pairing.",
          [&](unsigned x, unsigned y) { return detail::scalar_color(detail::sample(*ui, x, y)[3], 0, 1); },
          {{"source_artifact", "ui_source_color"}, {"channel", "A"}, {"black", 0}, {"white", 1},
            {"semantic", "consumed_ui_alpha"}, {"range", scalar_range(*ui, 3).describe()}});
      }
    }
    range depth_range;
    if (depth) {
      depth_range = scalar_range(*depth);
      const auto orientation = metadata.value("consumed_depth", json::object()).value("orientation", 0u);
      std::string caption = "Full raw allocation before crop and jitter. Black = minimum finite raw value; white = maximum; constant valid depth = gray. This is raw device depth, not linear distance. ";
      caption += orientation == 2 ? "Producer reports reversed depth (near has larger raw values)." : orientation == 1 ? "Producer reports forward depth (far has larger raw values)." :
                                                                                                                         "Depth orientation unknown; no near/far interpretation inferred.";
      json extra = {{"range", depth_range.describe()}, {"producer_orientation", orientation}};
      std::array<std::uint64_t, 32> bins {};
      for (unsigned y = 0; y < depth->height; ++y) {
        for (unsigned x = 0; x < depth->width; ++x) {
          const float v = detail::sample(*depth, x, y)[0];
          if (std::isfinite(v)) {
            ++bins[depth_range.high > depth_range.low ? std::min(31u, static_cast<unsigned>(std::max(0.0f, (v - depth_range.low) / (depth_range.high - depth_range.low)) * 32)) : 16];
          }
        }
      }
      extra["histogram_32_linear_raw_bins"] = bins;
      add("raw_depth.png", "Raw depth before cropping", depth->width, depth->height, caption, [&](unsigned x, unsigned y) {
        return detail::scalar_color(detail::sample(*depth, x, y)[0], depth_range.low, depth_range.high);
      },
          extra);
    }
    if (sbs) {
      add("final_sbs.png", "Final SBS (left eye | right eye)", sbs->width, sbs->height, std::string("Native renderer output before ReShade overlay, Windows cursor and host encoding. Alpha ignored for display. ") + (color_space == 2 || color_space == 3 ? hdr_note : encoded_note), [&](unsigned x, unsigned y) {
        return detail::display_color(detail::sample(*sbs, x, y), color_space == 3 ? 2 : color_space);
      });
    }
    for (const auto &entry : sunshine_game3d::ui_resources::catalog) {
      const auto *image = find(entry.file_stem);
      if (!image) continue;
      try {
        const auto captured = captured_resource(metadata, entry.artifact_id);
        const unsigned channels = channel_count(image->dxgi_format);
        const bool color_resource = entry.content == sunshine_game3d::ui_resources::role::color ||
          entry.content == sunshine_game3d::ui_resources::role::color_alpha;
        json extra = {{"artifact_id", entry.artifact_id}, {"semantic", entry.semantic},
          {"capture", captured}, {"allocation", "full; no depth crop or jitter applied"}};
        const std::string title = std::string(entry.source == sunshine_game3d::ui_resources::provider::streamline ? "SL " : "NGX ") + entry.name;
        const std::string filename = std::string(entry.file_stem) + ".png";
        if (color_resource && channels >= 3) {
          unsigned transfer = 0;
          const auto transfer_status = captured.find("transfer_status");
          if (transfer_status != captured.end() && (*transfer_status == "declared" || *transfer_status == "assumed") &&
              captured.contains("color_space") && captured["color_space"].is_number_integer() &&
              captured["color_space"] >= 1 && captured["color_space"] <= 3)
            transfer = captured["color_space"].get<unsigned>();
          if (transfer > 3) transfer = 0;
          const std::string transfer_note = transfer ? "Captured transfer " + captured["transfer_status"].get<std::string>() +
            "; " + (transfer == 1 ? std::string("encoded SDR codes displayed") : hdr_note) :
            "Unknown transfer: RGB code values displayed unchanged apart from 8-bit display clipping; final-color transfer is not assumed.";
          add(filename.c_str(), title + " RGB", image->width, image->height, transfer_note +
            " Full allocation; no crop, alignment or premultiplication correction. Alpha shown separately when present. This does not prove UI or frame matching.",
            [&](unsigned x, unsigned y) { return detail::display_color(detail::sample(*image, x, y), transfer); }, extra);
          if (channels == 4) {
            const auto alpha = std::string(entry.file_stem) + "_alpha.png";
            auto alpha_extra = extra;
            alpha_extra.update({{"channel", "A"}, {"black", 0}, {"white", 1}, {"range", scalar_range(*image, 3).describe()}});
            add(alpha.c_str(), title + " alpha", image->width, image->height,
              "Native A channel: 0 black, 1 white; display-only values outside [0,1] are clipped, nonfinite magenta. No alpha or UI coverage is inferred.",
              [&](unsigned x, unsigned y) { return detail::scalar_color(detail::sample(*image, x, y)[3], 0, 1); }, alpha_extra);
          }
        } else {
          constexpr const char *names[] {"R", "G", "B", "A"};
          for (unsigned channel = 0; channel < channels; ++channel) {
            auto component = extra;
            const auto bounds = scalar_range(*image, channel);
            component.update({{"channel", names[channel]}, {"range", bounds.describe()}});
            const bool motion = entry.content == sunshine_game3d::ui_resources::role::motion;
            const std::string file = channel ? std::string(entry.file_stem) + "_" + names[channel] + ".png" : filename;
            if (motion) component["symmetric_limit"] = bounds.signed_limit();
            else component.update({{"black", 0}, {"white", 1}});
            add(file.c_str(), title + " " + names[channel], image->width, image->height,
              motion ? "Native motion component, full allocation. Blue negative, orange positive, gray zero; symmetric full finite range. Units and alignment are not inferred." :
                "Native component, full allocation: 0 black, 1 white; display-only values outside [0,1] are clipped, nonfinite magenta. Constant zero and one remain distinct. No depth crop, mask threshold, or UI classification is applied.",
              [&](unsigned x, unsigned y) {
                const float value = detail::sample(*image, x, y)[channel];
                return motion ? detail::signed_color(value, bounds.signed_limit()) : detail::scalar_color(value, 0, 1);
              }, component);
          }
        }
      } catch (const std::exception &error) {
        report["optional_errors"].push_back({{"kind", entry.file_stem}, {"reason", error.what()}});
      }
    }
    if (color && depth && parameters.contains("depth_rect") && parameters.contains("jitter_uv")) {
      const auto rect = parameters.at("depth_rect").get<std::array<float, 4>>();
      const auto jitter = parameters.at("jitter_uv").get<std::array<float, 2>>();
      const bool valid = std::all_of(rect.begin(), rect.end(), [](float v) {
                           return std::isfinite(v);
                         }) &&
                         std::all_of(jitter.begin(), jitter.end(), [](float v) {
                           return std::isfinite(v);
                         }) &&
                         rect[0] >= 0 && rect[1] >= 0 && rect[2] > 0 && rect[3] > 0 && rect[0] + rect[2] <= 1.00001f && rect[1] + rect[3] <= 1.00001f;
      if (valid) {
        auto aligned = [&](unsigned x, unsigned y) {
          const auto p = detail::aligned_depth_pixel(x, y, color->width, color->height, depth->width, depth->height, rect, jitter);
          return detail::sample(*depth, p[0], p[1])[0];
        };
        add("depth.png", "Depth aligned to color", color->width, color->height, "Same point-sampling UV transform as the captured renderer: source UV * depth_rect extent + offset + jitter, clamped to active pixel centers. Same raw range as raw_depth.png; allocation padding is excluded. White means higher raw value, not necessarily farther away.", [&](unsigned x, unsigned y) {
          return detail::scalar_color(aligned(x, y), depth_range.low, depth_range.high);
        },
            {{"depth_rect", rect}, {"jitter_uv", jitter}});
        add("depth_alignment.png", "Depth contours over color", color->width, color->height, "Cyan = a raw-depth contour crossing (16 full-range bands) or a neighbor jump > 2% of full finite raw range. Cyan should follow visible object boundaries; nonlinear raw-depth spacing makes distant contours sparse. Magenta = nonfinite depth. This diagnostic is not a temporal alignment guarantee.", [&](unsigned x, unsigned y) {
          const float value = aligned(x, y);
          if (!std::isfinite(value)) {
            return invalid_color;
          }
          auto output = detail::display_color(detail::sample(*color, x, y), color_space);
          const float extent = depth_range.high - depth_range.low;
          if (!(extent > 0)) {
            return output;
          }
          const float v = (value - depth_range.low) / extent;
          const float a = (aligned(std::min(x + 1, color->width - 1), y) - depth_range.low) / extent;
          const float b = (aligned(x, std::min(y + 1, color->height - 1)) - depth_range.low) / extent;
          if (!std::isfinite(a) || !std::isfinite(b)) {
            return invalid_color;
          }
          if (std::abs(v - a) > 0.02f || std::abs(v - b) > 0.02f || std::floor(v * 16) != std::floor(a * 16) || std::floor(v * 16) != std::floor(b * 16)) {
            output = {static_cast<std::uint8_t>(output[0] / 4), static_cast<std::uint8_t>(191 + output[1] / 4), static_cast<std::uint8_t>(191 + output[2] / 4)};
          }
          return output;
        });
      } else {
        report["missing"].push_back("depth alignment: invalid crop/jitter parameters");
      }
    } else {
      report["missing"].push_back("depth alignment: missing color, depth or crop/jitter parameters");
    }
    if (field) {
      const auto extent = scalar_range(*field);
      add("signed_parallax.png", "Final signed parallax", field->width, field->height, "Blue = negative source-U; orange = positive; neutral gray = zero. Symmetric range uses maximum absolute finite displacement; sign is preserved. The warp samples right eye at u + field and left eye at u - field. Reported pixels describe the one-eye source-coordinate offset, not measured image error.", [&](unsigned x, unsigned y) {
        return detail::signed_color(detail::sample(*field, x, y)[0], extent.signed_limit());
      },
          {{"range_source_u", extent.describe()}, {"symmetric_limit_source_u", extent.signed_limit()}, {"symmetric_limit_source_pixels", extent.signed_limit() * field->width}, {"minimum_source_pixels", extent.finite ? json(extent.low * field->width) : json(nullptr)}, {"maximum_source_pixels", extent.finite ? json(extent.high * field->width) : json(nullptr)}});
    }
    if (field && candidate && field->width == candidate->width && field->height == candidate->height) {
      auto delta = [&](unsigned x, unsigned y) {
        return detail::sample(*field, x, y)[0] - detail::sample(*candidate, x, y)[0];
      };
      range extent;
      for (unsigned y = 0; y < field->height; ++y) {
        for (unsigned x = 0; x < field->width; ++x) {
          extent.add(delta(x, y));
        }
      }
      add("conditioning_delta.png", "Spatial conditioning: final minus candidate", field->width, field->height, "Blue = decreased signed displacement; orange = increased; neutral gray = unchanged. A display-only symmetric full range shows where spatial envelopes changed the warp. It does not judge whether conditioning is beneficial.", [&](unsigned x, unsigned y) {
        return detail::signed_color(delta(x, y), extent.signed_limit());
      },
          {{"range_source_u", extent.describe()}, {"symmetric_limit_source_pixels", extent.signed_limit() * field->width}});
    }
    if (sbs && sbs->width % 2 == 0) {
      const unsigned eye_width = sbs->width / 2;
      add("eye_difference.png", "Left/right difference (expected disparity)", eye_width, sbs->height, "Absolute RGB difference between display-mapped left and right eye at the same pixel, amplified 4x. Bright silhouettes are expected stereo disparity and occlusion; this is not a reconstruction error metric. Magenta marks nonfinite input.", [&](unsigned x, unsigned y) {
        const auto a = detail::sample(*sbs, x, y), b = detail::sample(*sbs, x + eye_width, y);
        for (unsigned c = 0; c < 3; ++c) {
          if (!std::isfinite(a[c]) || !std::isfinite(b[c])) {
            return invalid_color;
          }
        }
        const auto left = detail::display_color(a, color_space == 3 ? 2 : color_space), right = detail::display_color(b, color_space == 3 ? 2 : color_space);
        rgb result;
        for (unsigned c = 0; c < 3; ++c) {
          result[c] = static_cast<std::uint8_t>(std::min(255, 4 * std::abs(int(left[c]) - int(right[c]))));
        }
        return result;
      });
    }
    if (images.empty()) {
      report["status"] = "unavailable";
    }
    std::ofstream html(directory / "index.html", std::ios::binary);
    html.exceptions(std::ios::failbit | std::ios::badbit);
    html << "<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Game 3D dump</title><style>body{margin:32px auto;padding:0 20px;max-width:1400px;background:#111820;color:#e8edf3;font:16px/1.5 system-ui}a{color:#80caff}figure{margin:32px 0;padding:20px;background:#1d2835;border-radius:12px}img{width:100%;height:auto;background:#000}pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#b7c5d5}h2{margin-top:0}.legend{padding:16px;border-left:4px solid #80caff}</style><h1>Game 3D dump</h1><p class=\"legend\">Full-size PNGs for inspection. Click an image for its native resolution. Raw .bin files and <a href=\"manifest.json\">manifest.json</a> remain the replay source. Magenta marks NaN/Infinity. Blue = negative, orange = positive, gray = zero in signed maps. No percentile clipping is applied.</p>";
    json summary = {{"dump_status", manifest.value("status", json(nullptr))}, {"reason", manifest.value("reason", json(nullptr))}, {"missing_previews", report["missing"]}};
    for (const auto *key : {"consumed_depth", "render_parameters", "render_scene_policy", "pairing_evidence", "ui_resources", "optional_captures"}) {
      if (metadata.contains(key)) {
        summary[key] = metadata[key];
      }
    }
    summary["optional_capture_errors"] = manifest.value("optional_capture_errors", json::array());
    summary["optional_preview_errors"] = report["optional_errors"];
    const auto source = metadata.value("consumed_depth", json::object());
    html << "<p><strong>Source:</strong> " << escape(source.value("provider", "unknown")) << " &nbsp; <strong>FG:</strong> " << (source.value("frame_generation_active", false) ? "on" : "off") << " &nbsp; <strong>Reused depth:</strong> " << (source.value("reused_depth", false) ? "yes" : "no") << " &nbsp; <strong>Strength:</strong> " << escape(parameters.value("strength", json(nullptr)).dump()) << "% &nbsp; <strong>Scale:</strong> " << escape(parameters.value("depth_scale", json(nullptr)).dump()) << "</p>";
    html << "<p><strong>Capture status:</strong> " << escape(manifest.value("status", "unknown")) << " &nbsp; <strong>Color:</strong> " << (color_space == 1 ? "SDR sRGB" : color_space == 2 ? "HDR scRGB" :
                                                                                                                                                                          color_space == 3   ? "HDR PQ / Rec.2020" :
                                                                                                                                                                                               "unknown transfer")
         << " &nbsp; <strong>Missing:</strong> " << escape(report["missing"].dump()) << "</p>";
    html << "<details><summary>Complete source, geometry and frame association metadata</summary><pre>" << escape(summary.dump(2)) << "</pre></details>";
    for (const auto &entry : report["images"]) {
      const auto file = entry.at("file").get<std::string>();
      html << "<figure><h2>" << escape(entry.at("title").get<std::string>()) << "</h2><p>" << entry.at("width") << " &times; " << entry.at("height") << "</p><a href=\"" << file << "\"><img loading=\"lazy\" src=\"" << file << "\" alt=\"" << escape(entry.at("title").get<std::string>()) << "\"></a><figcaption>" << escape(entry.at("caption").get<std::string>()) << "</figcaption>";
      if (entry.contains("minimum_source_pixels")) {
        html << "<p><strong>One-eye displacement range:</strong> " << entry.at("minimum_source_pixels") << " to " << entry.at("maximum_source_pixels") << " source pixels</p>";
      } else if (entry.contains("symmetric_limit_source_pixels")) {
        html << "<p><strong>Symmetric display limit:</strong> +/- " << entry.at("symmetric_limit_source_pixels") << " source pixels</p>";
      }
      auto extra = entry;
      for (const auto *key : {"file", "title", "width", "height", "caption"}) {
        extra.erase(key);
      }
      if (!extra.empty()) {
        html << "<details><summary>Scale and statistics</summary><pre>" << escape(extra.dump(2)) << "</pre></details>";
      }
      html << "</figure>";
    }
    html << "</html>\n";
    return report;
  }
}  // namespace platf::game3d_debug::preview

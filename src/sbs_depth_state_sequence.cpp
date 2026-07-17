#include "sbs_depth_state_sequence.h"

#ifdef _WIN32

  #include "crypto.h"

  #include <algorithm>
  #include <array>
  #include <bit>
  #include <cctype>
  #include <cmath>
  #include <cstdio>
  #include <cstring>
  #include <exception>
  #include <fstream>
  #include <iterator>
  #include <limits>
  #include <nlohmann/json.hpp>
  #include <set>
  #include <system_error>

using Microsoft::WRL::ComPtr;

namespace sbs_bench::depth_state {
  namespace fs = std::filesystem;

  namespace {
    constexpr char manifest_name[] = "depth_state_manifest.json";
    constexpr char payload_directory[] = "frames";

    std::string sha256_hex(std::string_view value) {
      static constexpr char hex[] = "0123456789abcdef";
      const auto digest = crypto::hash(value);
      std::string result;
      result.reserve(digest.size() * 2u);
      for (const std::uint8_t byte : digest) {
        result.push_back(hex[byte >> 4]);
        result.push_back(hex[byte & 0x0f]);
      }
      return result;
    }

    bool sha256_file(const fs::path &path, std::string &result) {
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        return false;
      }
      const std::string contents(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
      );
      if (!stream.eof() && stream.fail()) {
        return false;
      }
      result = sha256_hex(contents);
      return true;
    }

    bool is_sha256(const std::string &value) {
      return value.size() == 64u &&
             std::all_of(value.begin(), value.end(), [](const unsigned char character) {
               return std::isdigit(character) ||
                      (character >= static_cast<unsigned char>('a') &&
                       character <= static_cast<unsigned char>('f'));
             });
    }

    bool json_unsigned_bounded(
      const nlohmann::json &object,
      std::string_view name,
      std::uint64_t maximum,
      std::uint64_t &value
    ) {
      if (!object.is_object()) {
        return false;
      }
      const auto field = object.find(std::string(name));
      if (field == object.end() || !field->is_number_unsigned()) {
        return false;
      }
      value = field->get<std::uint64_t>();
      return value <= maximum;
    }

    bool json_exact_unsigned_array(
      const nlohmann::json &value,
      const std::vector<std::uint64_t> &expected
    ) {
      if (!value.is_array() || value.size() != expected.size()) {
        return false;
      }
      for (std::size_t index = 0; index < expected.size(); ++index) {
        if (!value[index].is_number_unsigned() || value[index].get<std::uint64_t>() != expected[index]) {
          return false;
        }
      }
      return true;
    }

    std::string frame_directory(std::size_t ordinal) {
      char result[32];
      std::snprintf(result, sizeof(result), "frame_%06zu", ordinal);
      return result;
    }

    bool write_bytes(const fs::path &path, const void *data, std::size_t bytes) {
      std::ofstream stream(path, std::ios::binary | std::ios::trunc);
      if (!stream) {
        return false;
      }
      stream.write(static_cast<const char *>(data), static_cast<std::streamsize>(bytes));
      return static_cast<bool>(stream);
    }

    bool read_bytes(const fs::path &path, std::vector<std::uint8_t> &result) {
      std::error_code error;
      const auto size = fs::file_size(path, error);
      if (error || size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        return false;
      }
      result.resize(static_cast<std::size_t>(size));
      std::ifstream stream(path, std::ios::binary);
      if (!stream) {
        return false;
      }
      if (!result.empty()) {
        stream.read(reinterpret_cast<char *>(result.data()), static_cast<std::streamsize>(result.size()));
      }
      return static_cast<bool>(stream) || (result.empty() && stream.eof());
    }

    nlohmann::json file_identity(const fs::path &root, const fs::path &path) {
      std::string digest;
      if (!sha256_file(path, digest)) {
        return {};
      }
      std::error_code error;
      const auto size = fs::file_size(path, error);
      if (error) {
        return {};
      }
      return {
        {"path", fs::relative(path, root).generic_string()},
        {"bytes", size},
        {"sha256", digest},
      };
    }

    bool selected_ids_valid(
      const std::vector<std::uint64_t> &source,
      const std::vector<std::uint64_t> &selected
    ) {
      return !source.empty() && !selected.empty() &&
             std::is_sorted(source.begin(), source.end()) &&
             std::adjacent_find(source.begin(), source.end()) == source.end() &&
             std::is_sorted(selected.begin(), selected.end()) &&
             std::adjacent_find(selected.begin(), selected.end()) == selected.end() &&
             std::all_of(selected.begin(), selected.end(), [&source](const auto value) {
               return std::binary_search(source.begin(), source.end(), value);
             });
    }

    bool plain_existing_path_chain(const fs::path &path) {
      std::error_code error;
      const auto absolute = fs::absolute(path, error).lexically_normal();
      if (error || absolute.empty()) {
        return false;
      }
      fs::path current = absolute.root_path();
      for (const auto &part : absolute.relative_path()) {
        if (part.empty() || part == "." || part == "..") {
          return false;
        }
        current /= part;
        const DWORD attributes = GetFileAttributesW(current.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
          return false;
        }
      }
      return true;
    }

    bool read_texture_tight(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      ID3D11ShaderResourceView *view,
      DXGI_FORMAT expected_format,
      ComPtr<ID3D11Texture2D> &staging,
      std::vector<std::uint8_t> &bytes,
      UINT &width,
      UINT &height
    ) {
      if (!device || !context || !view) {
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      view->GetResource(&resource);
      ComPtr<ID3D11Texture2D> texture;
      if (FAILED(resource.As(&texture)) || !texture) {
        return false;
      }
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      if (!desc.Width || !desc.Height || desc.Format != expected_format || desc.MipLevels != 1u || desc.ArraySize != 1u) {
        return false;
      }
      bool recreate = !staging;
      if (!recreate) {
        D3D11_TEXTURE2D_DESC existing {};
        staging->GetDesc(&existing);
        recreate = existing.Width != desc.Width || existing.Height != desc.Height ||
                   existing.Format != desc.Format;
      }
      if (recreate) {
        auto stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        staging.Reset();
        if (FAILED(device->CreateTexture2D(&stage_desc, nullptr, &staging)) || !staging) {
          return false;
        }
      }
      context->CopyResource(staging.Get(), texture.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        return false;
      }
      constexpr std::size_t pixel_bytes = sizeof(std::uint32_t);
      bytes.resize(static_cast<std::size_t>(desc.Width) * desc.Height * pixel_bytes);
      for (UINT y = 0; y < desc.Height; ++y) {
        std::memcpy(
          bytes.data() + static_cast<std::size_t>(y) * desc.Width * pixel_bytes,
          static_cast<const std::uint8_t *>(mapped.pData) +
            static_cast<std::size_t>(y) * mapped.RowPitch,
          static_cast<std::size_t>(desc.Width) * pixel_bytes
        );
      }
      context->Unmap(staging.Get(), 0);
      width = desc.Width;
      height = desc.Height;
      return true;
    }

    bool read_raw_buffer(
      ID3D11Device *device,
      ID3D11DeviceContext *context,
      ID3D11ShaderResourceView *view,
      UINT width,
      UINT height,
      ComPtr<ID3D11Buffer> &staging,
      std::vector<std::uint8_t> &bytes,
      std::string &reason
    ) {
      if (!device || !context || !view || !width || !height) {
        reason = "missing device/context/view/dimensions";
        return false;
      }
      ComPtr<ID3D11Resource> resource;
      view->GetResource(&resource);
      ComPtr<ID3D11Buffer> buffer;
      if (FAILED(resource.As(&buffer)) || !buffer) {
        reason = "source SRV is not a buffer";
        return false;
      }
      D3D11_BUFFER_DESC desc {};
      buffer->GetDesc(&desc);
      const std::uint64_t required = static_cast<std::uint64_t>(width) * height * sizeof(float);
      // TensorRT/D3D interop rounds the backing buffer up to its allocation
      // alignment.  The authenticated raw tensor is only the logical H*W float
      // prefix, matching the established evaluator dump contract.
      if (required > desc.ByteWidth || desc.ByteWidth % sizeof(float) != 0u || desc.StructureByteStride != sizeof(float) || !(desc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)) {
        reason = "layout differs (required=" + std::to_string(required) +
                 ", bytes=" + std::to_string(desc.ByteWidth) +
                 ", stride=" + std::to_string(desc.StructureByteStride) +
                 ", misc=" + std::to_string(desc.MiscFlags) + ")";
        return false;
      }
      bool recreate = !staging;
      if (!recreate) {
        D3D11_BUFFER_DESC existing {};
        staging->GetDesc(&existing);
        recreate = existing.ByteWidth != desc.ByteWidth;
      }
      if (recreate) {
        auto stage_desc = desc;
        stage_desc.Usage = D3D11_USAGE_STAGING;
        stage_desc.BindFlags = 0;
        stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        stage_desc.MiscFlags = 0;
        stage_desc.StructureByteStride = 0;
        staging.Reset();
        const HRESULT created = device->CreateBuffer(&stage_desc, nullptr, &staging);
        if (FAILED(created) || !staging) {
          reason = "cannot create staging buffer (HRESULT=" +
                   std::to_string(static_cast<std::int64_t>(created)) + ")";
          return false;
        }
      }
      context->CopyResource(staging.Get(), buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      const HRESULT mapped_result =
        context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
      if (FAILED(mapped_result)) {
        reason = "cannot map staging buffer (HRESULT=" +
                 std::to_string(static_cast<std::int64_t>(mapped_result)) + ")";
        return false;
      }
      bytes.resize(static_cast<std::size_t>(required));
      std::memcpy(bytes.data(), mapped.pData, bytes.size());
      context->Unmap(staging.Get(), 0);
      reason.clear();
      return true;
    }

    bool safe_relative_payload_path(const fs::path &root, const std::string &relative, fs::path &resolved) {
      const fs::path rel(relative);
      if (relative.empty() || rel.is_absolute()) {
        return false;
      }
      for (const auto &part : rel) {
        if (part == "." || part == ".." || part.empty()) {
          return false;
        }
      }
      const auto unresolved = root / rel;
      if (!plain_existing_path_chain(root) || !plain_existing_path_chain(unresolved)) {
        return false;
      }
      std::error_code error;
      const auto normalized_root = fs::canonical(root, error);
      if (error || fs::is_symlink(root, error) || error) {
        return false;
      }
      resolved = fs::canonical(unresolved, error);
      if (error) {
        return false;
      }
      auto root_it = normalized_root.begin();
      auto path_it = resolved.begin();
      for (; root_it != normalized_root.end(); ++root_it, ++path_it) {
        if (path_it == resolved.end() || *root_it != *path_it) {
          return false;
        }
      }
      fs::path current = normalized_root;
      for (; path_it != resolved.end(); ++path_it) {
        current /= *path_it;
        if (fs::is_symlink(current, error) || error) {
          return false;
        }
        const DWORD attributes = GetFileAttributesW(current.c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_REPARSE_POINT)) {
          return false;
        }
      }
      return true;
    }

    bool read_authenticated_payload(
      const fs::path &root,
      const nlohmann::json &identity,
      std::uint64_t expected_bytes,
      std::vector<std::uint8_t> &bytes
    ) {
      if (!identity.is_object()) {
        return false;
      }
      const auto relative = identity.value("path", "");
      std::uint64_t recorded_bytes = 0;
      const auto recorded_digest = identity.value("sha256", "");
      fs::path path;
      if (!json_unsigned_bounded(identity, "bytes", std::numeric_limits<std::uint64_t>::max(), recorded_bytes) || recorded_bytes != expected_bytes || !is_sha256(recorded_digest) || !safe_relative_payload_path(root, relative, path) || !read_bytes(path, bytes) || bytes.size() != expected_bytes) {
        return false;
      }
      return sha256_hex(std::string_view(reinterpret_cast<const char *>(bytes.data()), bytes.size())) == recorded_digest;
    }
  }  // namespace

  struct sequence_writer::impl {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    fs::path root;
    std::string cache_key;
    std::vector<std::uint64_t> source_ids;
    std::vector<std::uint64_t> selected_ids;
    nlohmann::json frames = nlohmann::json::array();
    std::set<std::uint64_t> captured_selected;
    ComPtr<ID3D11Texture2D> depth_stage;
    ComPtr<ID3D11Texture2D> ema_stage;
    ComPtr<ID3D11Buffer> raw_stage;
    std::string failure;
    std::string manifest_digest;
    bool finished = false;

    impl(ComPtr<ID3D11Device> d, ComPtr<ID3D11DeviceContext> c, fs::path output, std::string key, std::vector<std::uint64_t> source, std::vector<std::uint64_t> selected):
        device(std::move(d)),
        context(std::move(c)),
        root(std::move(output)),
        cache_key(std::move(key)),
        source_ids(std::move(source)),
        selected_ids(std::move(selected)) {
      if (!device || !context || !is_sha256(cache_key) || !selected_ids_valid(source_ids, selected_ids)) {
        failure = "invalid depth-state export identity";
        return;
      }
      std::error_code error;
      if (fs::exists(root, error)) {
        if (error || !fs::is_directory(root) || !fs::is_empty(root)) {
          failure = "depth-state export root must be absent or empty";
          return;
        }
      } else if (!fs::create_directories(root, error) || error) {
        failure = "cannot create depth-state export root";
        return;
      }
      if (!fs::create_directories(root / payload_directory, error) || error) {
        failure = "cannot create depth-state frame directory";
      }
    }

    bool capture_scene(std::size_t ordinal, std::uint64_t source_id, const models::runtime_scene_evidence &evidence) {
      if (!failure.empty() || finished || ordinal != frames.size() || ordinal >= source_ids.size() || source_ids[ordinal] != source_id || !evidence.valid || evidence.completed_frame_id != ordinal) {
        failure = "depth-state runtime-scene sequence differs";
        return false;
      }
      frames.push_back({
        {"source_frame_ordinal", ordinal},
        {"source_frame_id", source_id},
        {"runtime_scene_id", evidence.runtime_scene_id},
        {"scene_age_float32_bits", std::bit_cast<std::uint32_t>(evidence.scene_age)},
        {"subject_initialized", evidence.subject_initialized},
        {"hard_cut", evidence.hard_cut},
        {"scene_start", evidence.scene_start},
        {"state", nullptr},
      });
      return true;
    }

    bool capture_state(std::size_t ordinal, std::uint64_t source_id, const models::estimate_result &estimate) {
      if (!failure.empty() || finished || ordinal >= frames.size() || source_ids[ordinal] != source_id || !std::binary_search(selected_ids.begin(), selected_ids.end(), source_id) || !captured_selected.insert(source_id).second || !estimate.depth || !estimate.subject || !estimate.raw_model_depth || !estimate.ema_motion_mask || estimate.raw_width <= 0 || estimate.raw_height <= 0) {
        failure = "selected depth-state frame is invalid or duplicated";
        return false;
      }
      const fs::path directory = root / payload_directory / frame_directory(ordinal);
      std::error_code directory_error;
      if (!fs::create_directories(directory, directory_error) || directory_error) {
        failure = "cannot create selected depth-state directory";
        return false;
      }
      std::vector<std::uint8_t> depth;
      std::vector<std::uint8_t> ema;
      std::vector<std::uint8_t> raw;
      UINT depth_width = 0, depth_height = 0, ema_width = 0, ema_height = 0;
      if (!read_texture_tight(device.Get(), context.Get(), estimate.depth.Get(), DXGI_FORMAT_R32_FLOAT, depth_stage, depth, depth_width, depth_height)) {
        failure = "cannot read exact selected depth texture";
        return false;
      }
      if (!read_texture_tight(device.Get(), context.Get(), estimate.ema_motion_mask.Get(), DXGI_FORMAT_R32_UINT, ema_stage, ema, ema_width, ema_height)) {
        failure = "cannot read exact selected EMA mask";
        return false;
      }
      if (depth_width != ema_width || depth_height != ema_height) {
        failure = "selected depth/EMA dimensions differ";
        return false;
      }
      std::string raw_reason;
      if (!read_raw_buffer(device.Get(), context.Get(), estimate.raw_model_depth.Get(), static_cast<UINT>(estimate.raw_width), static_cast<UINT>(estimate.raw_height), raw_stage, raw, raw_reason)) {
        failure = "cannot read exact selected raw-model depth: " + raw_reason;
        return false;
      }

      // The authoritative subject snapshot was already read synchronously by the caller.
      // It is supplied through the estimate's SubjectState SRV only indirectly, so capture it
      // from a staging buffer here rather than reconstructing any fields.
      ComPtr<ID3D11Resource> subject_resource;
      estimate.subject->GetResource(&subject_resource);
      ComPtr<ID3D11Buffer> subject_buffer;
      if (FAILED(subject_resource.As(&subject_buffer)) || !subject_buffer) {
        failure = "selected SubjectState is not a buffer";
        return false;
      }
      D3D11_BUFFER_DESC subject_desc {};
      subject_buffer->GetDesc(&subject_desc);
      if (subject_desc.ByteWidth != 12u * sizeof(float) || subject_desc.StructureByteStride != 4u * sizeof(float)) {
        failure = "selected SubjectState layout differs";
        return false;
      }
      auto stage_desc = subject_desc;
      stage_desc.Usage = D3D11_USAGE_STAGING;
      stage_desc.BindFlags = 0;
      stage_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      stage_desc.MiscFlags = 0;
      stage_desc.StructureByteStride = 0;
      ComPtr<ID3D11Buffer> subject_stage;
      if (FAILED(device->CreateBuffer(&stage_desc, nullptr, &subject_stage)) || !subject_stage) {
        failure = "cannot create SubjectState staging buffer";
        return false;
      }
      context->CopyResource(subject_stage.Get(), subject_buffer.Get());
      D3D11_MAPPED_SUBRESOURCE mapped {};
      if (FAILED(context->Map(subject_stage.Get(), 0, D3D11_MAP_READ, 0, &mapped))) {
        failure = "cannot map SubjectState staging buffer";
        return false;
      }
      std::array<float, 12> subject {};
      std::memcpy(subject.data(), mapped.pData, sizeof(subject));
      context->Unmap(subject_stage.Get(), 0);
      if (!std::all_of(subject.begin(), subject.end(), [](const float value) {
            return std::isfinite(value);
          })) {
        failure = "selected SubjectState contains a non-finite value";
        return false;
      }

      const fs::path depth_path = directory / "depth.r32f";
      const fs::path raw_path = directory / "raw.r32f";
      const fs::path ema_path = directory / "ema.r32u";
      const fs::path subject_path = directory / "subject.f32";
      if (!write_bytes(depth_path, depth.data(), depth.size()) || !write_bytes(raw_path, raw.data(), raw.size()) || !write_bytes(ema_path, ema.data(), ema.size()) || !write_bytes(subject_path, subject.data(), sizeof(subject))) {
        failure = "cannot write selected depth-state resources";
        return false;
      }
      const auto depth_identity = file_identity(root, depth_path);
      const auto raw_identity = file_identity(root, raw_path);
      const auto ema_identity = file_identity(root, ema_path);
      const auto subject_identity = file_identity(root, subject_path);
      if (depth_identity.empty() || raw_identity.empty() || ema_identity.empty() || subject_identity.empty()) {
        failure = "cannot authenticate selected depth-state resources";
        return false;
      }
      frames.at(ordinal)["state"] = {
        {"depth_width", depth_width},
        {"depth_height", depth_height},
        {"raw_width", estimate.raw_width},
        {"raw_height", estimate.raw_height},
        {"depth", depth_identity},
        {"raw", raw_identity},
        {"ema", ema_identity},
        {"subject", subject_identity},
      };
      return true;
    }

    bool finish_sequence(bool graph_captured) {
      if (!failure.empty() || finished || frames.size() != source_ids.size() || captured_selected.size() != selected_ids.size()) {
        failure = "depth-state sequence is incomplete";
        return false;
      }
      nlohmann::json selected_from_frames = nlohmann::json::array();
      for (const auto &frame : frames) {
        if (!frame.at("state").is_null()) {
          selected_from_frames.push_back(frame.at("source_frame_id"));
        }
      }
      if (selected_from_frames != nlohmann::json(selected_ids)) {
        failure = "depth-state selected-frame coverage differs";
        return false;
      }
      const nlohmann::json manifest {
        {"schema", sequence_schema},
        {"contract", sequence_contract},
        {"scope", "offline-ordinal-selected-frame-depth-state"},
        {"cache_key_sha256", cache_key},
        {"boundary", "completed-production-depth-state-before-warp-prefilter"},
        {"geometry_independence",
         "no-artistic-policy-plus-exact-scale-override-1-depth-state-v1"},
        {"source_frame_ids", source_ids},
        {"selected_frame_ids", selected_ids},
        {"source_frame_count", source_ids.size()},
        {"selected_frame_count", selected_ids.size()},
        {"cuda_graph_captured", graph_captured},
        {"frames", frames},
      };
      const fs::path path = root / manifest_name;
      const fs::path temporary = root / (std::string(manifest_name) + ".tmp");
      {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output << manifest.dump(2) << '\n';
        if (!output) {
          failure = "cannot write depth-state manifest";
          return false;
        }
      }
      std::error_code rename_error;
      fs::rename(temporary, path, rename_error);
      if (rename_error || !sha256_file(path, manifest_digest)) {
        failure = "cannot publish depth-state manifest";
        return false;
      }
      finished = true;
      return true;
    }
  };

  sequence_writer::sequence_writer(
    ComPtr<ID3D11Device> device,
    ComPtr<ID3D11DeviceContext> context,
    fs::path root,
    std::string cache_key,
    std::vector<std::uint64_t> source_frame_ids,
    std::vector<std::uint64_t> selected_frame_ids
  ):
      pimpl(std::make_unique<impl>(std::move(device), std::move(context), std::move(root), std::move(cache_key), std::move(source_frame_ids), std::move(selected_frame_ids))) {}

  sequence_writer::~sequence_writer() = default;

  bool sequence_writer::valid() const {
    return pimpl && pimpl->failure.empty();
  }

  const std::string &sequence_writer::error() const {
    return pimpl->failure;
  }

  bool sequence_writer::capture_runtime_scene(
    std::size_t ordinal,
    std::uint64_t source_id,
    const models::runtime_scene_evidence &evidence
  ) {
    return pimpl->capture_scene(ordinal, source_id, evidence);
  }

  bool sequence_writer::capture_selected_state(
    std::size_t ordinal,
    std::uint64_t source_id,
    const models::estimate_result &estimate
  ) {
    return pimpl->capture_state(ordinal, source_id, estimate);
  }

  bool sequence_writer::finish(bool graph_captured) {
    return pimpl->finish_sequence(graph_captured);
  }

  const std::string &sequence_writer::manifest_sha256() const {
    return pimpl->manifest_digest;
  }

  struct sequence_reader::impl {
    struct selected_payload {
      std::vector<std::uint8_t> depth;
      std::vector<std::uint8_t> raw;
      std::vector<std::uint8_t> ema;
      std::vector<std::uint8_t> subject;
    };

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    fs::path root;
    std::vector<std::uint64_t> source_ids;
    std::vector<std::uint64_t> selected_ids;
    nlohmann::json frames;
    std::vector<selected_payload> authenticated_payloads;
    std::string failure;
    std::string manifest_digest;
    bool graph_captured = false;
    ComPtr<ID3D11Texture2D> depth_texture;
    ComPtr<ID3D11ShaderResourceView> depth_srv;
    ComPtr<ID3D11Texture2D> ema_texture;
    ComPtr<ID3D11ShaderResourceView> ema_srv;
    ComPtr<ID3D11Buffer> raw_buffer;
    ComPtr<ID3D11ShaderResourceView> raw_srv;
    ComPtr<ID3D11Buffer> subject_buffer;
    ComPtr<ID3D11ShaderResourceView> subject_srv;

    impl(ComPtr<ID3D11Device> d, fs::path input, std::string expected_key, std::string expected_manifest, std::vector<std::uint64_t> expected_source, std::vector<std::uint64_t> expected_selected):
        device(std::move(d)),
        root(std::move(input)),
        source_ids(std::move(expected_source)),
        selected_ids(std::move(expected_selected)) {
      if (!device || !is_sha256(expected_key) || !is_sha256(expected_manifest) || !selected_ids_valid(source_ids, selected_ids)) {
        failure = "invalid depth-state replay identity";
        return;
      }
      fs::path path;
      std::vector<std::uint8_t> manifest_bytes;
      std::error_code size_error;
      if (!safe_relative_payload_path(root, manifest_name, path) || fs::file_size(path, size_error) > 64u * 1024u * 1024u || size_error || !read_bytes(path, manifest_bytes)) {
        failure = "cannot read authenticated depth-state replay manifest";
        return;
      }
      manifest_digest = sha256_hex(std::string_view(reinterpret_cast<const char *>(manifest_bytes.data()), manifest_bytes.size()));
      if (manifest_digest != expected_manifest) {
        failure = "depth-state replay manifest differs from outer CAS receipt";
        return;
      }
      nlohmann::json manifest;
      try {
        manifest = nlohmann::json::parse(
          manifest_bytes.begin(),
          manifest_bytes.end()
        );
      } catch (const std::exception &) {
        failure = "cannot parse depth-state replay manifest";
        return;
      }
      try {
        std::uint64_t schema_value = 0;
        std::uint64_t source_count = 0;
        std::uint64_t selected_count = 0;
        if (!manifest.is_object() || !json_unsigned_bounded(manifest, "schema", static_cast<std::uint64_t>(sequence_schema), schema_value) || schema_value != static_cast<std::uint64_t>(sequence_schema) || manifest.value("contract", "") != sequence_contract || manifest.value("scope", "") != "offline-ordinal-selected-frame-depth-state" || manifest.value("cache_key_sha256", "") != expected_key || manifest.value("boundary", "") != "completed-production-depth-state-before-warp-prefilter" || manifest.value("geometry_independence", "") != "no-artistic-policy-plus-exact-scale-override-1-depth-state-v1" || !manifest.contains("source_frame_ids") || !json_exact_unsigned_array(manifest["source_frame_ids"], source_ids) || !manifest.contains("selected_frame_ids") || !json_exact_unsigned_array(manifest["selected_frame_ids"], selected_ids) || !json_unsigned_bounded(manifest, "source_frame_count", source_ids.size(), source_count) || source_count != source_ids.size() || !json_unsigned_bounded(manifest, "selected_frame_count", selected_ids.size(), selected_count) || selected_count != selected_ids.size() || !manifest.contains("frames") || !manifest["frames"].is_array() || manifest["frames"].size() != source_ids.size()) {
          failure = "depth-state replay manifest identity differs";
          return;
        }
        frames = manifest["frames"];
        authenticated_payloads.resize(frames.size());
        graph_captured = manifest.value("cuda_graph_captured", false);

        std::set<std::string> seen_paths;
        std::vector<std::uint64_t> observed_selected;
        for (std::size_t ordinal = 0; ordinal < frames.size(); ++ordinal) {
          const auto &frame = frames[ordinal];
          std::uint64_t source_ordinal = 0;
          std::uint64_t source_id = 0;
          std::uint64_t runtime_scene_id = 0;
          std::uint64_t scene_age_bits = 0;
          if (!frame.is_object() || !json_unsigned_bounded(frame, "source_frame_ordinal", ordinal, source_ordinal) || source_ordinal != ordinal || !json_unsigned_bounded(frame, "source_frame_id", source_ids[ordinal], source_id) || source_id != source_ids[ordinal] || !json_unsigned_bounded(frame, "runtime_scene_id", std::numeric_limits<std::uint64_t>::max(), runtime_scene_id) || !json_unsigned_bounded(frame, "scene_age_float32_bits", std::numeric_limits<std::uint32_t>::max(), scene_age_bits) || !frame.contains("subject_initialized") || !frame["subject_initialized"].is_boolean() || !frame.contains("hard_cut") || !frame["hard_cut"].is_boolean() || !frame.contains("scene_start") || !frame["scene_start"].is_boolean() || !frame.contains("state")) {
            failure = "depth-state runtime-scene rows differ";
            return;
          }
          if (frame["state"].is_null()) {
            if (std::binary_search(selected_ids.begin(), selected_ids.end(), source_ids[ordinal])) {
              failure = "selected depth-state payload is missing";
              return;
            }
            continue;
          }
          if (!std::binary_search(selected_ids.begin(), selected_ids.end(), source_ids[ordinal]) || !frame["state"].is_object()) {
            failure = "unselected depth-state payload is present";
            return;
          }
          observed_selected.push_back(source_ids[ordinal]);
          const auto &state = frame["state"];
          auto &payload = authenticated_payloads[ordinal];
          std::uint64_t depth_width_value = 0;
          std::uint64_t depth_height_value = 0;
          std::uint64_t raw_width_value = 0;
          std::uint64_t raw_height_value = 0;
          if (!json_unsigned_bounded(state, "depth_width", 8192u, depth_width_value) || !json_unsigned_bounded(state, "depth_height", 8192u, depth_height_value) || !json_unsigned_bounded(state, "raw_width", 8192u, raw_width_value) || !json_unsigned_bounded(state, "raw_height", 8192u, raw_height_value) || !depth_width_value || !depth_height_value || !raw_width_value || !raw_height_value) {
            failure = "depth-state dimensions are invalid";
            return;
          }
          const auto depth_width = static_cast<UINT>(depth_width_value);
          const auto depth_height = static_cast<UINT>(depth_height_value);
          const auto raw_width = static_cast<UINT>(raw_width_value);
          const auto raw_height = static_cast<UINT>(raw_height_value);
          for (const auto &[name, expected_bytes] : std::array<std::pair<const char *, std::uint64_t>, 4> {{
                 {"depth", static_cast<std::uint64_t>(depth_width) * depth_height * 4u},
                 {"raw", static_cast<std::uint64_t>(raw_width) * raw_height * 4u},
                 {"ema", static_cast<std::uint64_t>(depth_width) * depth_height * 4u},
                 {"subject", 12u * sizeof(float)},
               }}) {
            if (!state.contains(name) || !state[name].is_object()) {
              failure = "depth-state file identity is missing";
              return;
            }
            const auto &identity = state[name];
            const auto relative = identity.value("path", "");
            auto *authenticated =
              std::string_view(name) == "depth" ? &payload.depth :
              std::string_view(name) == "raw"   ? &payload.raw :
              std::string_view(name) == "ema"   ? &payload.ema :
                                                  &payload.subject;
            if (!seen_paths.insert(relative).second || !read_authenticated_payload(root, identity, expected_bytes, *authenticated)) {
              failure = "depth-state payload bytes differ";
              return;
            }
          }
        }
        if (observed_selected != selected_ids) {
          failure = "depth-state selected payload coverage differs";
        }
      } catch (const std::exception &) {
        failure = "depth-state replay manifest types differ";
      }
    }

    bool scene(std::size_t ordinal, std::uint64_t source_id, models::runtime_scene_evidence &result) const {
      if (!failure.empty() || ordinal >= frames.size() || source_ids[ordinal] != source_id) {
        return false;
      }
      const auto &frame = frames[ordinal];
      result.valid = true;
      result.completed_frame_id = ordinal;
      result.runtime_scene_id = frame.at("runtime_scene_id").get<std::uint64_t>();
      result.scene_age = std::bit_cast<float>(
        frame.at("scene_age_float32_bits").get<std::uint32_t>()
      );
      result.subject_initialized = frame.at("subject_initialized").get<bool>();
      result.hard_cut = frame.at("hard_cut").get<bool>();
      result.scene_start = frame.at("scene_start").get<bool>();
      if (!frame.at("state").is_null()) {
        const auto &subject = authenticated_payloads[ordinal].subject;
        if (subject.size() != sizeof(result.subject_state)) {
          return false;
        }
        std::memcpy(result.subject_state.data(), subject.data(), subject.size());
        if (!std::all_of(
              result.subject_state.begin(),
              result.subject_state.end(),
              [](const float value) {
                return std::isfinite(value);
              }
            )) {
          return false;
        }
      }
      return std::isfinite(result.scene_age);
    }

    bool load(std::size_t ordinal, std::uint64_t source_id, models::estimate_result &result) {
      if (!failure.empty() || ordinal >= frames.size() || source_ids[ordinal] != source_id || !std::binary_search(selected_ids.begin(), selected_ids.end(), source_id)) {
        return false;
      }
      const auto &state = frames[ordinal].at("state");
      const UINT depth_width = state.at("depth_width").get<UINT>();
      const UINT depth_height = state.at("depth_height").get<UINT>();
      const UINT raw_width = state.at("raw_width").get<UINT>();
      const UINT raw_height = state.at("raw_height").get<UINT>();
      const auto &payload = authenticated_payloads[ordinal];
      const auto &depth = payload.depth;
      const auto &raw = payload.raw;
      const auto &ema = payload.ema;
      const auto &subject = payload.subject;
      const std::uint64_t depth_bytes =
        static_cast<std::uint64_t>(depth_width) * depth_height * 4u;
      const std::uint64_t raw_bytes =
        static_cast<std::uint64_t>(raw_width) * raw_height * 4u;
      if (depth.size() != depth_bytes || raw.size() != raw_bytes || ema.size() != depth_bytes || subject.size() != 12u * sizeof(float)) {
        return false;
      }

      D3D11_TEXTURE2D_DESC texture_desc {};
      texture_desc.Width = depth_width;
      texture_desc.Height = depth_height;
      texture_desc.MipLevels = 1;
      texture_desc.ArraySize = 1;
      texture_desc.Format = DXGI_FORMAT_R32_FLOAT;
      texture_desc.SampleDesc.Count = 1;
      texture_desc.Usage = D3D11_USAGE_IMMUTABLE;
      texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      D3D11_SUBRESOURCE_DATA texture_data {depth.data(), depth_width * 4u, 0};
      depth_texture.Reset();
      depth_srv.Reset();
      if (FAILED(device->CreateTexture2D(&texture_desc, &texture_data, &depth_texture)) || FAILED(device->CreateShaderResourceView(depth_texture.Get(), nullptr, &depth_srv)) || !depth_srv) {
        return false;
      }
      texture_desc.Format = DXGI_FORMAT_R32_UINT;
      D3D11_SUBRESOURCE_DATA ema_data {ema.data(), depth_width * 4u, 0};
      ema_texture.Reset();
      ema_srv.Reset();
      if (FAILED(device->CreateTexture2D(&texture_desc, &ema_data, &ema_texture)) || FAILED(device->CreateShaderResourceView(ema_texture.Get(), nullptr, &ema_srv)) || !ema_srv) {
        return false;
      }

      D3D11_BUFFER_DESC raw_desc {};
      raw_desc.Usage = D3D11_USAGE_IMMUTABLE;
      raw_desc.ByteWidth = static_cast<UINT>(raw.size());
      raw_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      raw_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      raw_desc.StructureByteStride = sizeof(float);
      D3D11_SUBRESOURCE_DATA raw_data {raw.data(), 0, 0};
      raw_buffer.Reset();
      raw_srv.Reset();
      if (FAILED(device->CreateBuffer(&raw_desc, &raw_data, &raw_buffer)) || FAILED(device->CreateShaderResourceView(raw_buffer.Get(), nullptr, &raw_srv)) || !raw_srv) {
        return false;
      }

      D3D11_BUFFER_DESC subject_desc {};
      subject_desc.Usage = D3D11_USAGE_IMMUTABLE;
      subject_desc.ByteWidth = static_cast<UINT>(subject.size());
      subject_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
      subject_desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
      subject_desc.StructureByteStride = 4u * sizeof(float);
      D3D11_SUBRESOURCE_DATA subject_data {subject.data(), 0, 0};
      subject_buffer.Reset();
      subject_srv.Reset();
      if (FAILED(device->CreateBuffer(&subject_desc, &subject_data, &subject_buffer)) || FAILED(device->CreateShaderResourceView(subject_buffer.Get(), nullptr, &subject_srv)) || !subject_srv) {
        return false;
      }

      result = {};
      result.depth = depth_srv;
      result.subject = subject_srv;
      result.ema_motion_mask = ema_srv;
      result.raw_model_depth = raw_srv;
      result.raw_width = static_cast<int>(raw_width);
      result.raw_height = static_cast<int>(raw_height);
      result.completed_frame_valid = true;
      result.completed_frame_id = ordinal;
      result.cuda_graph_active = graph_captured;
      return true;
    }
  };

  sequence_reader::sequence_reader(
    ComPtr<ID3D11Device> device,
    fs::path root,
    std::string expected_cache_key,
    std::string expected_manifest_sha256,
    std::vector<std::uint64_t> expected_source_frame_ids,
    std::vector<std::uint64_t> expected_selected_frame_ids
  ):
      pimpl(std::make_unique<impl>(std::move(device), std::move(root), std::move(expected_cache_key), std::move(expected_manifest_sha256), std::move(expected_source_frame_ids), std::move(expected_selected_frame_ids))) {}

  sequence_reader::~sequence_reader() = default;

  bool sequence_reader::valid() const {
    return pimpl && pimpl->failure.empty();
  }

  const std::string &sequence_reader::error() const {
    return pimpl->failure;
  }

  bool sequence_reader::runtime_scene(
    std::size_t ordinal,
    std::uint64_t source_id,
    models::runtime_scene_evidence &evidence
  ) const {
    return pimpl->scene(ordinal, source_id, evidence);
  }

  bool sequence_reader::load_selected_state(
    std::size_t ordinal,
    std::uint64_t source_id,
    models::estimate_result &estimate
  ) {
    return pimpl->load(ordinal, source_id, estimate);
  }

  bool sequence_reader::cuda_graph_captured() const {
    return pimpl->graph_captured;
  }

  const std::string &sequence_reader::manifest_sha256() const {
    return pimpl->manifest_digest;
  }

}  // namespace sbs_bench::depth_state

#endif  // _WIN32

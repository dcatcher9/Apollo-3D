// SPDX-License-Identifier: GPL-3.0-only
#include "game3d_debug_dump.h"

#include "game3d_debug_preview.h"
#include "src/game3d_debug_ui_resources.h"
#include "src/game3d_debug_formats.h"
#include "sbs_debug_dump_async.h"
#include "src/config.h"
#include "src/logging.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <d3d11_1.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>
#include <vector>
#include <wrl/client.h>

namespace platf::game3d_debug {
  namespace {
    namespace wire = ::game3d_debug;
    using Microsoft::WRL::ComPtr;
    using json = nlohmann::json;
    using clock_t = std::chrono::steady_clock;
    constexpr std::size_t readback_bytes_per_poll = 8 * 1024 * 1024;
    constexpr auto request_timeout = std::chrono::seconds(15);

    std::uint64_t read64(std::uint64_t &value) {
      return static_cast<std::uint64_t>(InterlockedCompareExchange64(
        reinterpret_cast<volatile LONG64 *>(&value),
        0,
        0
      ));
    }

    void write64(std::uint64_t &value, std::uint64_t next) {
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&value), static_cast<LONG64>(next));
    }

    std::uint64_t creation_time(HANDLE process) {
      FILETIME created {}, exited {}, kernel {}, user {};
      return GetProcessTimes(process, &created, &exited, &kernel, &user) ?
               (static_cast<std::uint64_t>(created.dwHighDateTime) << 32) | created.dwLowDateTime :
               0;
    }

    std::uint64_t unique_id() {
      static std::atomic<std::uint64_t> next {[] {
        LARGE_INTEGER now {};
        QueryPerformanceCounter(&now);
        return static_cast<std::uint64_t>(now.QuadPart) ^ (static_cast<std::uint64_t>(GetCurrentProcessId()) << 32);
      }()};
      const auto id = next.fetch_add(1, std::memory_order_relaxed);
      return id ? id : next.fetch_add(1, std::memory_order_relaxed);
    }

    class handle_t {
    public:
      explicit handle_t(HANDLE value = nullptr):
          value_(value) {}

      ~handle_t() {
        reset();
      }

      handle_t(const handle_t &) = delete;
      handle_t &operator=(const handle_t &) = delete;

      HANDLE get() const {
        return value_;
      }

      void reset(HANDLE value = nullptr) {
        if (value_) {
          CloseHandle(value_);
        }
        value_ = value;
      }

    private:
      HANDLE value_;
    };

    const char *artifact_name(wire::artifact kind) {
      switch (kind) {
        case wire::artifact::source_color:
          return "source_color";
        case wire::artifact::raw_depth:
          return "raw_depth";
        case wire::artifact::candidate:
          return "candidate";
        case wire::artifact::vertical_majorant:
          return "vertical_majorant";
        case wire::artifact::vertical_field:
          return "vertical_field";
        case wire::artifact::final_field:
          return "final_field";
        case wire::artifact::sbs:
          return "sbs";
        case wire::artifact::linear_color:
          return "linear_color";
        case wire::artifact::ui_source_color:
          return "ui_source_color";
        default:
          return sunshine_game3d::ui_resources::artifact_name(static_cast<unsigned>(kind));
      }
    }

    bool optional_artifact(wire::artifact kind) {
      return static_cast<unsigned>(kind) >= 9 && kind != wire::artifact::ui_source_color;
    }

    std::filesystem::path output_directory() {
      if (const char *value = std::getenv("APOLLO_SBS_DUMP"); value && *value) {
        return value;
      }
      return config::sunshine.log_file.empty() ? std::filesystem::path("sbs_dump") :
                                                 std::filesystem::path(config::sunshine.log_file).parent_path() / "sbs_dump";
    }

    struct image_t {
      wire::texture_t descriptor;
      std::vector<std::uint8_t> bytes;
    };

    struct package_t {
      json manifest;
      std::vector<image_t> images;
      std::string name;
    };

    void publish(const std::filesystem::path &root, const package_t &package) {
      const auto temporary = root / (package.name + ".tmp");
      const auto final = root / package.name;
      bool owns_temporary = false;
      bool raw_package_ready = false;
      try {
        std::filesystem::create_directories(root);
        if (!std::filesystem::create_directory(temporary)) {
          throw std::runtime_error("temporary directory already exists");
        }
        owns_temporary = true;
        auto manifest = package.manifest;
        manifest["artifacts"] = json::array();
        const auto write_metadata = [](const std::filesystem::path &path, const json &value) {
          std::ofstream metadata(path, std::ios::binary);
          metadata.exceptions(std::ios::badbit | std::ios::failbit);
          metadata << value.dump(2) << '\n';
          metadata.close();
        };
        manifest["visualizations"] = {{"status", "unavailable"}, {"reason", "Preview generation or metadata publication did not complete"}, {"raw_artifacts_preserved", true}};
        for (std::size_t i = 0; i < package.images.size(); ++i) {
          const auto &image = package.images[i];
          const auto &desc = image.descriptor;
          if (optional_artifact(desc.kind) && !raw_package_ready) {
            // Optional bytes must not consume the last disk space needed to
            // describe the already-written replay inputs. Preserve this base
            // manifest until a complete additive replacement is ready.
            write_metadata(temporary / "manifest.json", manifest);
            raw_package_ready = true;
          }
          const std::string file = std::to_string(i) + "_" + artifact_name(desc.kind) + ".bin";
          try {
            std::ofstream stream(temporary / file, std::ios::binary);
            stream.exceptions(std::ios::badbit | std::ios::failbit);
            stream.write(reinterpret_cast<const char *>(image.bytes.data()), image.bytes.size());
            stream.close();
          } catch (const std::exception &error) {
            if (!optional_artifact(desc.kind)) throw;
            std::error_code ignored;
            std::filesystem::remove(temporary / file, ignored);
            auto &errors = manifest["optional_capture_errors"];
            if (errors.is_null()) errors = json::array();
            errors.push_back({{"artifact_id", static_cast<unsigned>(desc.kind)}, {"kind", artifact_name(desc.kind)},
              {"stage", "write"}, {"reason", error.what()}});
            continue;
          }
          manifest["artifacts"].push_back({{"artifact_id", static_cast<unsigned>(desc.kind)}, {"kind", artifact_name(desc.kind)}, {"file", file}, {"width", desc.width}, {"height", desc.height}, {"dxgi_format", desc.dxgi_format}, {"row_bytes", desc.width * detail::bytes_per_pixel(desc.dxgi_format)}, {"byte_count", image.bytes.size()}, {"layout", "tightly packed rows, top to bottom, native little-endian DXGI pixels"}});
        }
        manifest["not_captured"] = json::array();
        const auto missing = [&](wire::artifact kind) {
          if (std::none_of(manifest["artifacts"].begin(), manifest["artifacts"].end(), [kind](const json &image) {
                return image.at("kind") == artifact_name(kind);
              })) {
            manifest["not_captured"].push_back(artifact_name(kind));
          }
        };
        for (const auto kind : {wire::artifact::source_color, wire::artifact::raw_depth, wire::artifact::candidate, wire::artifact::vertical_majorant, wire::artifact::vertical_field, wire::artifact::final_field, wire::artifact::sbs, wire::artifact::linear_color, wire::artifact::ui_source_color}) {
          missing(kind);
        }
        for (const auto &entry : sunshine_game3d::ui_resources::catalog) {
          missing(static_cast<wire::artifact>(entry.artifact_id));
        }
        // Commit a usable raw manifest before optional PNGs consume disk space.
        // This remains the valid fallback if preview generation or its metadata
        // update fails, including on a full disk.
        if (!raw_package_ready) write_metadata(temporary / "manifest.json", manifest);
        raw_package_ready = true;
        // This is the CPU publication worker, after GPU readback has finished.
        // Preview failure must never discard the lossless reproduction package.
        try {
          std::vector<preview::image_view> images;
          images.reserve(package.images.size());
          for (const auto &image : package.images) {
            const auto &desc = image.descriptor;
            if (std::none_of(manifest["artifacts"].begin(), manifest["artifacts"].end(), [&](const json &written) {
                  return written.at("kind") == artifact_name(desc.kind);
                })) continue;
            images.push_back({artifact_name(desc.kind), desc.width, desc.height, desc.dxgi_format, image.bytes});
          }
          manifest["visualizations"] = preview::generate(temporary, manifest, images);
        } catch (const std::exception &error) {
          manifest["visualizations"] = {{"status", "failed"}, {"reason", error.what()}, {"raw_artifacts_preserved", true}};
          BOOST_LOG(warning) << "Game 3D dump previews failed; preserving raw package: " << error.what();
        }
        const auto updated_metadata = temporary / "manifest.preview.tmp";
        try {
          write_metadata(updated_metadata, manifest);
          if (!MoveFileExW(updated_metadata.c_str(), (temporary / "manifest.json").c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            throw std::runtime_error("cannot publish preview metadata, Windows error " + std::to_string(GetLastError()));
          }
        } catch (const std::exception &error) {
          std::error_code ignored;
          std::filesystem::remove(updated_metadata, ignored);
          BOOST_LOG(warning) << "Game 3D dump preview metadata failed; preserving original raw manifest: " << error.what();
        }
        std::filesystem::rename(temporary, final);
        BOOST_LOG(info) << "Game 3D dump published: " << final.string();
      } catch (const std::exception &error) {
        if (owns_temporary && !raw_package_ready) {
          std::error_code ignored;
          std::filesystem::remove_all(temporary, ignored);
        }
        BOOST_LOG(warning) << "Game 3D dump publication failed: " << error.what();
        if (raw_package_ready) {
          BOOST_LOG(warning) << "Recoverable Game 3D raw package retained at " << temporary.string();
        }
      }
    }
  }  // namespace

  namespace detail {
    std::uint32_t bytes_per_pixel(std::uint32_t format) noexcept {
      return ::game3d_debug::pixel_bytes(format);
    }

    bool validate_response(const wire::response_t &response, std::uint64_t nonce, std::string &reason) {
      if (!nonce || response.consumer_nonce != nonce) {
        reason = "response nonce mismatch";
        return false;
      }
      if (response.result != wire::status::complete && response.result != wire::status::unavailable && response.result != wire::status::failed) {
        reason = "invalid producer result";
        return false;
      }
      if (response.texture_count > wire::max_textures || response.json_bytes > wire::max_json_bytes) {
        reason = "response bounds exceeded";
        return false;
      }
      if (response.result == wire::status::complete && (!response.capture_id || !response.texture_count)) {
        reason = "complete response has no captured textures";
        return false;
      }
      std::uint64_t total = 0;
      for (std::uint32_t i = 0; i < response.texture_count; ++i) {
        const auto &texture = response.textures[i];
        // Optional resources are validated and admitted independently below.
        // A newer catalog or unsupported mask format cannot lose replay inputs.
        if (optional_artifact(texture.kind)) continue;
        const auto bpp = bytes_per_pixel(texture.dxgi_format);
        if (!artifact_name(texture.kind) || !texture.handle || !bpp || !texture.width || !texture.height || texture.width > wire::max_dimension || texture.height > wire::max_dimension) {
          reason = "unsupported texture descriptor";
          return false;
        }
        for (std::uint32_t j = 0; j < i; ++j) {
          if (optional_artifact(response.textures[j].kind)) continue;
          if (response.textures[j].kind == texture.kind || response.textures[j].handle == texture.handle) {
            reason = "duplicate texture descriptor";
            return false;
          }
        }
        total += static_cast<std::uint64_t>(texture.width) * texture.height * bpp;
        if (total > wire::max_capture_bytes) {
          reason = "capture memory limit exceeded";
          return false;
        }
      }
      return true;
    }
  }  // namespace detail

  class dumper::impl_t {
  public:
    impl_t(observer_t observe, std::filesystem::path directory):
        observe_(std::move(observe)),
        directory_(directory.empty() ? output_directory() : std::move(directory)),
        publication_(sbs_debug::detail::publication_state::create()) {}

    ~impl_t() {
      cancel();
    }

    void set_button(std::shared_ptr<std::atomic<bool>> button) {
      button_ = std::move(button);
    }

    bool needed() const noexcept {
      return pending_ || (button_ && button_->load(std::memory_order_acquire) && !publication_->busy());
    }

    void cancel() noexcept {
      if (button_) {
        button_->store(false, std::memory_order_release);
      }
      retire();
      package_.reset();
    }

    void poll(ID3D11Device *device, ID3D11DeviceContext *context, RECT fullscreen, int width, int height, int mode) {
      try {
        if (!pending_) {
          if (publication_->busy() || !button_ || !button_->exchange(false, std::memory_order_acq_rel)) {
            return;
          }
          start(device, context, fullscreen, width, height, mode);
          return;
        }
        if (device != device_.Get() || context != context_.Get() || width != width_ || height != height_ || mode != mode_ || !EqualRect(&fullscreen, &fullscreen_)) {
          finish("unavailable", "presentation changed during capture");
          return;
        }
        if (clock_t::now() - started_ > request_timeout) {
          finish("unavailable", "snapshot timed out");
          return;
        }
        if (copy_done_) {
          readback();
          return;
        }
        if (WaitForSingleObject(process_.get(), 0) != WAIT_TIMEOUT) {
          finish("unavailable", "producer exited");
          return;
        }
        if (!identity_valid() || read64(shared_->consumer_nonce) != nonce_ || read64(shared_->request_id) != request_id_) {
          finish("unavailable", "producer or consumer identity changed");
          return;
        }
        if (read64(shared_->response_id) != request_id_) {
          return;
        }
        receive();
      } catch (const std::exception &error) {
        try {
          finish("failed", error.what());
        } catch (...) {
          // Diagnostics must not tear down streaming even if recording the failure allocates.
          retire();
          package_.reset();
        }
      }
    }

  private:
    bool identity_valid() const {
      return shared_ && shared_->signature == wire::magic && shared_->protocol_version == wire::version &&
             shared_->shared_bytes == sizeof(wire::shared_state_t) && shared_->producer_pid == producer_pid_ &&
             shared_->producer_creation_time == producer_creation_;
    }

    void acknowledge() noexcept {
      if (shared_ && request_id_ && read64(shared_->consumer_nonce) == nonce_ && read64(shared_->request_id) == request_id_) {
        write64(shared_->released_id, request_id_);
      }
    }

    void retire() noexcept {
      acknowledge();
      if (shared_) {
        UnmapViewOfFile(std::exchange(shared_, nullptr));
      }
      mapping_.reset();
      process_.reset();
      staging_.clear();
      retained_.clear();
      copy_done_.Reset();
      device_.Reset();
      context_.Reset();
      pending_ = false;
      image_index_ = 0;
      row_ = 0;
      request_id_ = 0;
    }

    void finish(const char *status, const std::string &reason) {
      if (!package_) {
        retire();
        return;
      }
      package_->manifest["status"] = status;
      package_->manifest["reason"] = reason;
      // Only completely read artifacts are published; partial CPU images have no valid layout.
      package_->images.erase(std::remove_if(package_->images.begin(), package_->images.end(), [](const image_t &image) {
                               return image.bytes.size() != static_cast<std::uint64_t>(image.descriptor.width) * image.descriptor.height *
                                                              detail::bytes_per_pixel(image.descriptor.dxgi_format);
                             }),
                             package_->images.end());
      auto package = std::move(package_);
      retire();
      const auto directory = directory_;
      if (!publication_->enqueue([directory, package] {
            publish(directory, *package);
          })) {
        BOOST_LOG(warning) << "Game 3D dump publication queue rejected the package";
      }
    }

    void start(ID3D11Device *device, ID3D11DeviceContext *context, RECT fullscreen, int width, int height, int mode) {
      nonce_ = unique_id();
      request_id_ = unique_id();
      package_ = std::make_shared<package_t>();
      package_->name = "game3d_" + std::to_string(GetCurrentProcessId()) + "_" + std::to_string(request_id_);
      LARGE_INTEGER frequency {};
      QueryPerformanceFrequency(&frequency);
      package_->manifest = {{"schema", "sunshine.game3d.dump.v1"}, {"request_id", request_id_}, {"consumer_nonce", nonce_}, {"consumer_pid", GetCurrentProcessId()}, {"consumer_creation_time", creation_time(GetCurrentProcess())}, {"source_width", width}, {"source_height", height}, {"wire_mode", mode}, {"qpc_ticks_per_second", frequency.QuadPart}, {"units", {{"raw_depth", "exact provider depth values; projection/conversion in producer_metadata"}, {"color", "native DXGI encoded values; no gamma, exposure, clipping, or tone mapping applied"}, {"fields", "native shader values; producer_metadata defines stage interpretation"}, {"capture_qpc", "QueryPerformanceCounter ticks"}, {"process_creation_time", "Windows FILETIME: 100ns ticks since 1601-01-01 UTC"}}}};
      if (!device || !context || width <= 0 || height <= 0 || (mode != 2 && mode != 3)) {
        finish("unavailable", "invalid Game presentation or D3D context");
        return;
      }
      ComPtr<ID3D11Device> context_device;
      context->GetDevice(&context_device);
      if (context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE || context_device.Get() != device) {
        finish("unavailable", "Game dump needs the capture device's immediate context");
        return;
      }
      const auto foreground = observe_();
      const auto rect = foreground.client_screen_rect;
      if (!foreground_window::carries_geometry(foreground.status) || !foreground.process_id || fullscreen.right <= fullscreen.left || fullscreen.bottom <= fullscreen.top || rect.left > fullscreen.left || rect.top > fullscreen.top || rect.right < fullscreen.right || rect.bottom < fullscreen.bottom) {
        finish("unavailable", "foreground game does not cover the captured display");
        return;
      }
      producer_pid_ = foreground.process_id;
      package_->manifest["producer_pid"] = producer_pid_;
      process_.reset(OpenProcess(PROCESS_DUP_HANDLE | PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, FALSE, producer_pid_));
      producer_creation_ = process_.get() ? creation_time(process_.get()) : 0;
      if (!producer_creation_) {
        finish("unavailable", "cannot verify producer process");
        return;
      }
      package_->manifest["producer_creation_time"] = producer_creation_;
      const auto name = std::wstring(wire::mapping_prefix) + std::to_wstring(producer_pid_);
      mapping_.reset(OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, name.c_str()));
      shared_ = mapping_.get() ? static_cast<wire::shared_state_t *>(MapViewOfFile(mapping_.get(), FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(wire::shared_state_t))) : nullptr;
      if (!identity_valid()) {
        finish("unavailable", "Game 3D diagnostic producer absent or incompatible");
        return;
      }
      device_ = device;
      context_ = context;
      fullscreen_ = fullscreen;
      width_ = width;
      height_ = height;
      mode_ = mode;
      started_ = clock_t::now();
      pending_ = true;
      shared_->consumer_pid = GetCurrentProcessId();
      shared_->consumer_creation_time = creation_time(GetCurrentProcess());
      write64(shared_->consumer_nonce, nonce_);
      write64(shared_->request_id, request_id_);
    }

    void receive() {
      wire::response_t response;
      std::memcpy(&response, &shared_->response, sizeof(response));
      std::string reason;
      if (!detail::validate_response(response, nonce_, reason)) {
        finish("failed", reason);
        return;
      }
      const std::string producer_json(shared_->json, response.json_bytes);
      if (read64(shared_->response_id) != request_id_ || read64(shared_->consumer_nonce) != nonce_) {
        finish("unavailable", "response changed while reading metadata");
        return;
      }
      const auto metadata = producer_json.empty() ? json::object() : json::parse(producer_json, nullptr, false);
      if (!metadata.is_object()) {
        finish("failed", "invalid producer JSON object");
        return;
      }
      package_->manifest["producer_metadata"] = metadata;
      package_->manifest["capture_id"] = response.capture_id;
      package_->manifest["capture_qpc"] = response.capture_qpc;
      package_->manifest["runtime_epoch"] = response.runtime_epoch;
      package_->manifest["export_generation"] = response.export_generation;
      package_->manifest["export_sequence"] = response.export_sequence;
      package_->manifest["producer_result"] = static_cast<std::uint32_t>(response.result);
      package_->manifest["producer_flags"] = response.flags;
      if (response.flags & wire::optional_metadata_omitted) {
        package_->manifest["optional_capture_errors"] = json::array({{
          {"stage", "producer_metadata"}, {"reason", "Optional capture metadata could not be serialized within the mailbox budget; primary data and its middleware inventory were preserved. Optional pixels were not published."}}});
      }
      response_status_ = response.result;
      if (!response.texture_count) {
        finish(response.result == wire::status::unavailable ? "unavailable" : "failed", "producer returned no textures; see producer_metadata");
        return;
      }
      ComPtr<ID3D11Device1> device1;
      if (FAILED(device_.As(&device1))) {
        finish("failed", "D3D11 shared NT handles unsupported");
        return;
      }
      std::uint64_t admitted_bytes = 0;
      // Admit replay inputs first even if a producer interleaves optional ones.
      for (const bool optional : {false, true}) {
        for (std::uint32_t i = 0; i < response.texture_count; ++i) {
          const auto &entry = response.textures[i];
          if (optional_artifact(entry.kind) != optional) continue;
          const auto rejected = [&](const char *reason) {
            if (optional) optional_error(entry, "open", reason);
            else finish("failed", reason);
            return optional;
          };
          const auto bpp = detail::bytes_per_pixel(entry.dxgi_format);
          const auto size = std::uint64_t(entry.width) * entry.height * bpp;
          if (!artifact_name(entry.kind) || !entry.handle || !bpp || !entry.width || !entry.height ||
              entry.width > wire::max_dimension || entry.height > wire::max_dimension ||
              size > wire::max_capture_bytes - admitted_bytes) {
            if (rejected("unsupported optional descriptor or capture byte budget exceeded")) continue;
            return;
          }
          if (std::any_of(package_->images.begin(), package_->images.end(), [&](const image_t &image) {
                return image.descriptor.kind == entry.kind || image.descriptor.handle == entry.handle;
              })) {
            if (rejected("duplicate snapshot descriptor")) continue;
            return;
          }
          HANDLE duplicated = nullptr;
          if (!DuplicateHandle(process_.get(), reinterpret_cast<HANDLE>(static_cast<std::uintptr_t>(entry.handle)), GetCurrentProcess(), &duplicated, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
            if (rejected("cannot duplicate snapshot handle")) continue;
            return;
          }
          handle_t duplicate(duplicated);
          ComPtr<ID3D11Texture2D> source;
          if (FAILED(device1->OpenSharedResource1(duplicate.get(), IID_PPV_ARGS(&source)))) {
            if (rejected("cannot open snapshot texture on capture device")) continue;
            return;
          }
          D3D11_TEXTURE2D_DESC desc {};
          source->GetDesc(&desc);
          if (desc.Width != entry.width || desc.Height != entry.height || desc.Format != static_cast<DXGI_FORMAT>(entry.dxgi_format) || desc.MipLevels != 1 || desc.ArraySize != 1 || desc.SampleDesc.Count != 1 || desc.SampleDesc.Quality != 0) {
            if (rejected("shared texture does not match snapshot descriptor")) continue;
            return;
          }
          desc.Usage = D3D11_USAGE_STAGING;
          desc.BindFlags = 0;
          desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
          desc.MiscFlags = 0;
          ComPtr<ID3D11Texture2D> staging;
          if (FAILED(device_->CreateTexture2D(&desc, nullptr, &staging))) {
            if (rejected("cannot allocate snapshot staging texture")) continue;
            return;
          }
          retained_.push_back(source);
          staging_.push_back(staging);
          package_->images.push_back({entry, {}});
          admitted_bytes += size;
        }
      }
      if (staging_.empty()) {
        finish("unavailable", "no readable snapshot textures; see optional_capture_errors");
        return;
      }
      // Immutable producer resources now have receiver COM owners; releasing NT handles is safe.
      acknowledge();
      D3D11_QUERY_DESC query {D3D11_QUERY_EVENT, 0};
      if (FAILED(device_->CreateQuery(&query, &copy_done_))) {
        finish("failed", "cannot create snapshot completion query");
        return;
      }
      for (std::size_t i = 0; i < staging_.size(); ++i) {
        context_->CopyResource(staging_[i].Get(), retained_[i].Get());
      }
      context_->End(copy_done_.Get());
      // Submit this one diagnostic batch even when the game/desktop stops presenting.
      context_->Flush();
    }

    void readback() {
      BOOL done = FALSE;
      const auto queried = context_->GetData(copy_done_.Get(), &done, sizeof(done), D3D11_ASYNC_GETDATA_DONOTFLUSH);
      if (queried == S_FALSE || (queried == S_OK && !done)) {
        return;
      }
      if (FAILED(queried)) {
        finish("failed", "snapshot GPU copy failed");
        return;
      }
      std::size_t budget = readback_bytes_per_poll;
      while (image_index_ < staging_.size() && budget) {
        auto &image = package_->images[image_index_];
        const auto row_bytes = image.descriptor.width * detail::bytes_per_pixel(image.descriptor.dxgi_format);
        const auto rows = std::min<std::size_t>(image.descriptor.height - row_, budget / row_bytes);
        if (!rows) {
          return;
        }
        D3D11_MAPPED_SUBRESOURCE mapped {};
        const auto result = context_->Map(staging_[image_index_].Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped);
        if (result == DXGI_ERROR_WAS_STILL_DRAWING) {
          return;
        }
        if (FAILED(result)) {
          if (skip_optional_readback("snapshot readback failed")) continue;
          finish("failed", "snapshot readback failed");
          return;
        }
        if (!mapped.pData || mapped.RowPitch < row_bytes) {
          context_->Unmap(staging_[image_index_].Get(), 0);
          if (skip_optional_readback("invalid snapshot row pitch")) continue;
          finish("failed", "invalid snapshot row pitch");
          return;
        }
        try {
          // Reserve avoids a full-image recopy on each bounded append. Only copied rows are initialized.
          if (image.bytes.empty()) {
            image.bytes.reserve(static_cast<std::size_t>(row_bytes) * image.descriptor.height);
          }
          for (std::size_t i = 0; i < rows; ++i) {
            const auto *source = static_cast<const std::uint8_t *>(mapped.pData) + (row_ + i) * mapped.RowPitch;
            image.bytes.insert(image.bytes.end(), source, source + row_bytes);
          }
        } catch (...) {
          context_->Unmap(staging_[image_index_].Get(), 0);
          if (skip_optional_readback("cannot allocate optional snapshot bytes")) continue;
          throw;
        }
        context_->Unmap(staging_[image_index_].Get(), 0);
        row_ += static_cast<std::uint32_t>(rows);
        budget -= rows * row_bytes;
        if (row_ == image.descriptor.height) {
          ++image_index_;
          row_ = 0;
        }
      }
      if (image_index_ == staging_.size()) {
        finish(response_status_ == wire::status::complete ? "complete" : "partial", response_status_ == wire::status::complete ? "" : "producer reported incomplete capture; see producer_metadata");
      }
    }

    void optional_error(const wire::texture_t &entry, const char *stage, const char *reason) {
      auto &errors = package_->manifest["optional_capture_errors"];
      if (errors.is_null()) errors = json::array();
      const char *name = artifact_name(entry.kind);
      errors.push_back({{"artifact_id", static_cast<unsigned>(entry.kind)}, {"kind", name ? json(name) : json(nullptr)},
        {"stage", stage}, {"reason", reason}});
    }

    bool skip_optional_readback(const char *reason) {
      auto &image = package_->images[image_index_];
      if (!optional_artifact(image.descriptor.kind)) return false;
      optional_error(image.descriptor, "readback", reason);
      image.bytes.clear();
      ++image_index_;
      row_ = 0;
      return true;
    }

    observer_t observe_;
    std::filesystem::path directory_;
    std::shared_ptr<sbs_debug::detail::publication_state> publication_;
    std::shared_ptr<std::atomic<bool>> button_;
    std::shared_ptr<package_t> package_;
    handle_t process_, mapping_;
    wire::shared_state_t *shared_ = nullptr;
    std::uint32_t producer_pid_ = 0;
    std::uint64_t producer_creation_ = 0, nonce_ = 0, request_id_ = 0;
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    std::vector<ComPtr<ID3D11Texture2D>> retained_, staging_;
    ComPtr<ID3D11Query> copy_done_;
    wire::status response_status_ = wire::status::failed;
    clock_t::time_point started_ {};
    RECT fullscreen_ {};
    int width_ = 0, height_ = 0, mode_ = 0;
    std::size_t image_index_ = 0;
    std::uint32_t row_ = 0;
    bool pending_ = false;
  };

  dumper::dumper(observer_t observe, std::filesystem::path directory):
      impl_(std::make_unique<impl_t>(std::move(observe), std::move(directory))) {}

  dumper::~dumper() = default;

  void dumper::set_button_request(std::shared_ptr<std::atomic<bool>> request) {
    impl_->set_button(std::move(request));
  }

  void dumper::poll(ID3D11Device *device, ID3D11DeviceContext *context, RECT fullscreen, int width, int height, int mode) {
    impl_->poll(device, context, fullscreen, width, height, mode);
  }

  void dumper::cancel() noexcept {
    impl_->cancel();
  }

  bool dumper::needs_conversion_poll() const noexcept {
    return impl_->needed();
  }
}  // namespace platf::game3d_debug

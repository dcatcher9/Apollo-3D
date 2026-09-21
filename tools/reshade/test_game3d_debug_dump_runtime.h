// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include "game3d_debug_dump.h"
#include "scene_gain.h"
#include "src/game3d_debug_protocol.h"

#include <d3d11_1.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <wrl/client.h>

namespace sunshine_game3d_test {
  namespace wire = ::game3d_debug;

  class dump_fixture {
    sunshine_game3d::debug_dump producer_;
    HANDLE mapping_ = nullptr;
    wire::shared_state_t *shared_ = nullptr;
    std::uint64_t request_ = 0;
    sunshine_game3d::diagnostic_frame frame_;
    Microsoft::WRL::ComPtr<ID3D11Device1> consumer_device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> consumer_context_;

    static void check(bool ok, const char *message) {
      if (!ok) {
        throw std::runtime_error(message);
      }
    }

    static void publish(std::uint64_t &field, std::uint64_t value) {
      InterlockedExchange64(reinterpret_cast<volatile LONG64 *>(&field), value);
    }

    void ensure_d3d11_consumer(reshade::api::effect_runtime *runtime) {
      if (consumer_device_) {
        return;
      }
      Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
      Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
      Microsoft::WRL::ComPtr<ID3D11Device> device;
      const auto luid = reinterpret_cast<ID3D12Device *>(runtime->get_device()->get_native())->GetAdapterLuid();
      check(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))), "Cannot select the D3D12 snapshot adapter for D3D11 reception");
      check(SUCCEEDED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &consumer_context_)) && SUCCEEDED(device.As(&consumer_device_)), "Cannot create an independent D3D11 diagnostic consumer");
    }

    std::vector<std::uint8_t> read_d3d11(ID3D11Texture2D *texture) {
      D3D11_TEXTURE2D_DESC desc {};
      texture->GetDesc(&desc);
      const unsigned bpp = desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT                                            ? 16 :
                           desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT || desc.Format == DXGI_FORMAT_R32G32_FLOAT ? 8 :
                                                                                                                      4;
      desc.Usage = D3D11_USAGE_STAGING;
      desc.BindFlags = desc.MiscFlags = 0;
      desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
      Microsoft::WRL::ComPtr<ID3D11Texture2D> staging;
      Microsoft::WRL::ComPtr<ID3D11Query> done;
      const D3D11_QUERY_DESC query {D3D11_QUERY_EVENT, 0};
      check(SUCCEEDED(consumer_device_->CreateTexture2D(&desc, nullptr, &staging)) && SUCCEEDED(consumer_device_->CreateQuery(&query, &done)), "Cannot stage the D3D12 snapshot through D3D11");
      consumer_context_->CopyResource(staging.Get(), texture);
      consumer_context_->End(done.Get());
      consumer_context_->Flush();
      const auto deadline = GetTickCount64() + 5000;
      HRESULT ready;
      while ((ready = consumer_context_->GetData(done.Get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH)) == S_FALSE && GetTickCount64() < deadline) {
        Sleep(1);
      }
      check(ready == S_OK, "D3D11 diagnostic reception did not complete");
      D3D11_MAPPED_SUBRESOURCE mapped {};
      check(SUCCEEDED(consumer_context_->Map(staging.Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &mapped)), "Cannot map completed D3D11 diagnostic reception");
      const size_t row = size_t(desc.Width) * bpp;
      std::vector<std::uint8_t> bytes(row * desc.Height);
      for (unsigned y = 0; y < desc.Height; ++y) {
        std::memcpy(bytes.data() + row * y, static_cast<const std::uint8_t *>(mapped.pData) + size_t(mapped.RowPitch) * y, row);
      }
      consumer_context_->Unmap(staging.Get(), 0);
      return bytes;
    }

  public:
    dump_fixture() {
      FILETIME created {}, exited {}, kernel {}, user {};
      check(GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user), "Cannot identify dump test process");
      const auto creation = (std::uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime;
      check(producer_.initialize(GetCurrentProcessId(), creation), "Cannot initialize real producer diagnostic mailbox");
      const auto name = std::wstring(wire::mapping_prefix) + std::to_wstring(GetCurrentProcessId());
      mapping_ = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, name.c_str());
      check(mapping_ != nullptr, "Cannot open diagnostic mailbox as a consumer");
      shared_ = static_cast<wire::shared_state_t *>(MapViewOfFile(mapping_, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(wire::shared_state_t)));
      check(shared_ != nullptr, "Cannot map diagnostic mailbox");
      shared_->consumer_pid = GetCurrentProcessId();
      shared_->consumer_creation_time = creation;
      shared_->consumer_nonce = 0x5342531234ULL;
      check(!producer_.requested(), "Idle diagnostic producer is armed");
    }

    ~dump_fixture() {
      if (shared_) {
        UnmapViewOfFile(shared_);
      }
      if (mapping_) {
        CloseHandle(mapping_);
      }
    }

    void begin(reshade::api::effect_runtime *runtime, sunshine_game3d::renderer &renderer, const sunshine_game3d::render_parameters &p, reshade::api::resource_view depth, bool reused, reshade::api::color_space color) {
      namespace api = reshade::api;
      frame_ = {};
      frame_.parameters = renderer.consumed_parameters();
      frame_.source_alpha_ui = renderer.consumed_source_alpha_ui();
      frame_.ui_plane = renderer.consumed_ui_plane();
      frame_.resources = renderer.diagnostics();
      frame_.source_alpha_decision.requested = frame_.source_alpha_ui;
      frame_.source_alpha_decision.retained_alpha_ready = frame_.resources.ui_source.handle != 0;
      if (frame_.resources.ui_source.handle) frame_.ui_source_metadata = R"({"fixture":true,"sequence":42,"same_game_frame":"unverified"})";
      frame_.shader_source = renderer.active_shader_source();
      frame_.backbuffer_index = runtime->get_current_back_buffer_index();
      frame_.presentation_ordinal = 1 + request_;
      frame_.color = color;
      frame_.export_generation = 43;
      frame_.export_sequence = 101 + request_;
      frame_.qpc = 123456 + request_;
      auto &d = frame_.depth;
      d.runtime_epoch = 37;
      d.frame_index = 91;
      d.source_id = 77;
      d.layout_epoch = 12;
      d.ready = p.depth_ready && depth.handle;
      d.reused_depth = reused;
      d.frame_generation_active = reused;
      d.provided.sequence = 71;
      d.provided.tick = 54321;
      d.provided.epoch = 13;
      // Supply distinct diagnostic targets to exercise metadata transport;
      // this renderer fixture does not run or impersonate the scene policy.
      auto &scale = frame_.scene_status.scale;
      scale.basis = p.camera_ready ? sunshine_game3d::automatic_scale_basis::camera_matrix :
        sunshine_game3d::automatic_scale_basis::relative_depth;
      scale.value = p.depth_scale > 0.f ? p.depth_scale : 1.f;
      scale.active = d.ready;
      scale.target_value = d.ready && !reused ? 2.f : 0.f;
      scale.has_zero = true;
      scale.zero_inverse = frame_.parameters.convergence[1];
      scale.minimum_inverse = reused ? .4 : 0.;
      scale.maximum_inverse = scale.reference_inverse = reused ? .4 : 2.;
      scale.mean_inverse = reused ? .4 : .3;
      scale.normalization = 4.;
      scale.has_depth_statistics = d.ready;
      scale.target_zero_inverse = reused ? .4 : 1.;
      scale.zero_target_available = d.ready;
      scale.ui_midpoint_inverse = frame_.ui_plane.inverse_depth;
      scale.has_ui_midpoint = true;
      scale.target_ui_midpoint_inverse = reused ? .4 : 1.;
      scale.ui_midpoint_target_available = d.ready;
      if (d.ready) {
        d.shader_resource = depth;
        d.resource = runtime->get_device()->get_resource_from_view(depth);
        d.source_resource = d.resource;
        const auto desc = runtime->get_device()->get_resource_desc(d.resource);
        d.width = desc.texture.width;
        d.height = desc.texture.height;
        d.active_width = d.width;
        d.active_height = d.height;
      }
      publish(shared_->released_id, 0);
      publish(shared_->request_id, ++request_);
      producer_.poll(true);
      check(!producer_.requested(), "Dump consumed the frame before arming its API observation window");
      const auto observation_deadline = GetTickCount64() + 250;
      while (!producer_.requested() && GetTickCount64() < observation_deadline) {
        Sleep(1);
        producer_.poll(true);
      }
      check(producer_.requested(), "Real dump request did not admit the bounded no-SDK observation window");
      producer_.capture(runtime, runtime->get_command_queue()->get_immediate_command_list(), frame_);
      producer_.poll();
      check(shared_->response_id != request_, "Diagnostic pixels published before submission fence");
      check(!producer_.requested(), "A second diagnostic capture can overwrite an in-flight one");
    }

    void submitted(reshade::api::effect_runtime *runtime) {
      producer_.finish_present(runtime->get_command_queue()->get_native(), runtime->get_native());
    }

    template<class Read>
    void verify(reshade::api::effect_runtime *runtime, Read read, const std::filesystem::path &output_directory = {}) {
      namespace api = reshade::api;
      producer_.poll();
      // ReShade's test drain need not make a separate native D3D11 fence
      // CPU-visible immediately. Follow the real consumer's asynchronous
      // contract without submitting another signal or bypassing completion.
      const auto completion_deadline = GetTickCount64() + 5000;
      while (shared_->response_id != request_ && GetTickCount64() < completion_deadline) {
        Sleep(1);
        producer_.poll();
      }
      check(shared_->response_id == request_, "Diagnostic snapshot did not publish within five seconds of submission");
      const auto response = shared_->response;
      check(response.result == wire::status::complete, "GPU diagnostic snapshot failed");
      check(response.runtime_epoch == 37 && response.export_generation == 43 && response.export_sequence == frame_.export_sequence, "Diagnostic render/export identity changed while pending");
      const auto json = nlohmann::json::parse(std::string(shared_->json, response.json_bytes));
      check(json.at("consumed_depth").at("provider_sequence") == 71 && json.at("consumed_depth").at("reused_depth") == frame_.depth.reused_depth && json.at("render_parameters").at("strength") == frame_.parameters.strength, "Dump lost its captured depth lineage or render parameters");
      check(json.at("overlay_included") == false, "Pre-overlay dump mislabeled");
      const auto &policy = json.at("render_scene_policy");
      const auto &scale = frame_.scene_status.scale;
      check(policy.at("zero_tracking").at("time_constant_seconds") == sunshine_camera_scene::time_constant_seconds &&
        policy.at("zero_tracking").at("zero_budget_per_second") == sunshine_scene_gain::zero_budget_per_second,
        "Dump zero tracking constants differ from the production policy");
      check(policy.at("zero_inverse") == scale.zero_inverse &&
        policy.at("applied_zero_inverse") == frame_.parameters.convergence[1],
        "Dump changed the current or rendered zero while publishing its target");
      check(policy.at("target_scale") == (scale.has_target() ? nlohmann::json(scale.target_value) : nlohmann::json(nullptr)),
        "Dump invented a gain target for held or positive-flat evidence");
      if (scale.has_zero_target()) {
        const bool camera = scale.basis == sunshine_game3d::automatic_scale_basis::camera_matrix;
        check(policy.at("target_zero_inverse") == scale.target_zero_inverse &&
          policy.at("target_zero_plane") == (camera ? 1. / scale.target_zero_inverse : scale.target_zero_inverse) &&
          policy.at("depth_statistics").at("mean_q") == scale.mean_inverse,
          "Dump lost the accepted midpoint target or confused it with the diagnostic mean");
      } else {
        check(policy.at("target_zero_inverse").is_null() && policy.at("target_zero_plane").is_null() &&
          policy.at("depth_statistics").is_null(), "Missing-depth dump borrowed held UI statistics or targets");
      }
      const auto &replay = json.at("replay");
      check(replay.at("source_alpha_ui") == frame_.source_alpha_ui &&
        replay.at("ui_constant_binding").at("uint32")[0] == (frame_.source_alpha_ui ? 1u : 0u),
        "Dump lost the actual source-alpha UI selection or its b1 binding");
      const bool external_ui = frame_.source_alpha_ui && frame_.resources.ui_source.handle;
      check(replay.at("ui_alpha_source") == (!frame_.source_alpha_ui ? "none" : external_ui ? "ui_source_color" : "source_color"),
        "Dump lost the actual consumed UI-alpha source identity");
      if (external_ui) check(json.at("ui_source").at("sequence") == 42, "Consumed UI provenance was not preserved");
      const auto hex = replay.at("parameter_hex").get<std::string>();
      check(replay.at("parameter_bytes") == sizeof(frame_.parameters) && hex.size() == sizeof(frame_.parameters) * 2, "Replay constants do not describe the complete ABI");
      const auto *constant_bytes = reinterpret_cast<const unsigned char *>(&frame_.parameters);
      constexpr char digits[] = "0123456789abcdef";
      const auto ui_words = sunshine_game3d::ui_parameter_words(frame_.source_alpha_ui, frame_.ui_plane);
      const auto ui_hex = replay.at("ui_parameter_hex").get<std::string>();
      const bool nearest_plane = frame_.ui_plane.mode == sunshine_game3d::ui_plane_mode::depth_midpoint_nearest_ui;
      const bool front_plane = frame_.ui_plane.mode == sunshine_game3d::ui_plane_mode::front_limit;
      check(replay.at("ui_parameter_abi") == (front_plane ? "sunshine_game3d.ui_parameters.v4" : nearest_plane ? "sunshine_game3d.ui_parameters.v3" : "sunshine_game3d.ui_parameters.v2") &&
          replay.at("ui_parameter_bytes") == sizeof(ui_words) && ui_hex.size() == sizeof(ui_words) * 2 &&
          replay.at("ui_constant_binding").at("uint32") == ui_words,
        "Dump lost exact consumed independent UI plane words");
      check(replay.at("ui_plane_resolution").at("reduction_ran") == bool(frame_.resources.ui_plane_resolved.handle) &&
          replay.at("ui_plane_resolution").at("inverse_depth_role") == (front_plane || frame_.ui_plane.mode == sunshine_game3d::ui_plane_mode::screen ? "unused" : nearest_plane ? "midpoint_floor" : "explicit_plane"),
        "Dump confused the submitted UI base with the GPU-resolved plane");
      const auto *ui_bytes = reinterpret_cast<const unsigned char *>(ui_words.data());
      for (size_t i = 0; i < sizeof(ui_words); ++i)
        check(ui_hex[i * 2] == digits[ui_bytes[i] >> 4] && ui_hex[i * 2 + 1] == digits[ui_bytes[i] & 15],
          "Replay UI constants differ from the exact GPU buffer bytes");
      for (size_t i = 0; i < sizeof(frame_.parameters); ++i) {
        check(hex[i * 2] == digits[constant_bytes[i] >> 4] && hex[i * 2 + 1] == digits[constant_bytes[i] & 15], "Replay constants differ from the exact GPU buffer bytes");
      }
      check(replay.at("shader_source") == frame_.shader_source && json.at("pairing_evidence").at("same_game_frame") == "unverified", "Replay source or temporal pairing evidence is incorrect");
      std::vector<Microsoft::WRL::ComPtr<IUnknown>> held;
      std::vector<Microsoft::WRL::ComPtr<ID3D11Texture2D>> held_d3d11;
      std::vector<std::vector<std::uint8_t>> bytes;
      const bool cross_api = runtime->get_device()->get_api() == api::device_api::d3d12;
      if (cross_api) {
        ensure_d3d11_consumer(runtime);
      }
      bool has_raw = false, has_source = false, has_sbs = false, has_ui_source = false;
      for (unsigned i = 0; i < response.texture_count; ++i) {
        const auto &t = response.textures[i];
        Microsoft::WRL::ComPtr<IUnknown> opened;
        if (runtime->get_device()->get_api() == api::device_api::d3d11) {
          Microsoft::WRL::ComPtr<ID3D11Device1> device;
          Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
          check(SUCCEEDED(reinterpret_cast<ID3D11Device *>(runtime->get_device()->get_native())->QueryInterface(IID_PPV_ARGS(&device))), "No D3D11 NT sharing interface");
          check(SUCCEEDED(device->OpenSharedResource1(reinterpret_cast<HANDLE>(t.handle), IID_PPV_ARGS(&texture))), "Cannot open immutable D3D11 snapshot");
          opened = texture;
        } else {
          Microsoft::WRL::ComPtr<ID3D12Resource> texture;
          check(SUCCEEDED(reinterpret_cast<ID3D12Device *>(runtime->get_device()->get_native())->OpenSharedHandle(reinterpret_cast<HANDLE>(t.handle), IID_PPV_ARGS(&texture))), "Cannot open immutable D3D12 snapshot");
          opened = texture;
        }
        api::resource original {};
        const auto &r = frame_.resources;
        switch (t.kind) {
          case wire::artifact::ui_source_color:
            original = r.ui_source;
            has_ui_source = true;
            break;
          case wire::artifact::source_color:
            original = r.source;
            has_source = true;
            break;
          case wire::artifact::sbs:
            original = r.sbs;
            has_sbs = true;
            break;
          case wire::artifact::linear_color:
            original = r.linear_color;
            break;
          case wire::artifact::candidate:
            original = r.candidate;
            break;
          case wire::artifact::vertical_majorant:
            original = r.vertical_majorant;
            break;
          case wire::artifact::vertical_field:
            original = r.vertical_field;
            break;
          case wire::artifact::final_field:
            original = r.final_field;
            break;
          case wire::artifact::raw_depth:
            original = frame_.depth.resource;
            has_raw = true;
            break;
        }
        check(original.handle != 0, "Snapshot fabricated an absent artifact");
        const auto actual = read(api::resource {reinterpret_cast<std::uint64_t>(opened.Get())}, true);
        check(actual == read(original, false), "Shared GPU snapshot differs from the exact consumed/rendered pixels");
        if (cross_api) {
          Microsoft::WRL::ComPtr<ID3D11Texture2D> received;
          check(SUCCEEDED(consumer_device_->OpenSharedResource1(reinterpret_cast<HANDLE>(t.handle), IID_PPV_ARGS(&received))), "Host D3D11 cannot open the immutable D3D12 diagnostic texture");
          check(read_d3d11(received.Get()) == actual, "Cross-API D3D12 to D3D11 snapshot bytes differ");
          held_d3d11.emplace_back(std::move(received));
        }
        held.emplace_back(std::move(opened));
        bytes.push_back(actual);
      }
      check(has_source && has_sbs && has_raw == frame_.depth.ready, "Diagnostic artifact availability is incorrect");
      check(has_ui_source == external_ui, "Diagnostic UI source availability differs from the exact renderer input");
      publish(shared_->released_id, request_);
      producer_.poll();
      // The consumer's COM references must preserve immutable pixels after the
      // producer closes every exported handle and releases its ownership.
      for (unsigned i = 0; i < held.size(); ++i) {
        check(read(api::resource {reinterpret_cast<std::uint64_t>(held[i].Get())}, true) == bytes[i], "Acknowledgment invalidated immutable diagnostic pixels");
      }
      for (unsigned i = 0; i < held_d3d11.size(); ++i) {
        check(read_d3d11(held_d3d11[i].Get()) == bytes[i], "Acknowledgment invalidated D3D11 consumer ownership of D3D12 snapshots");
      }
      check(!producer_.requested(), "Acknowledged request was captured again");
      if (!output_directory.empty()) {
        check(std::filesystem::create_directories(output_directory), "Diagnostic fixture output already exists");
        nlohmann::json manifest {{"schema", "sunshine.game3d.dump.v1"}, {"status", "complete"}, {"request_id", request_}, {"consumer_nonce", response.consumer_nonce}, {"producer_metadata", json}, {"capture_id", response.capture_id}, {"capture_qpc", response.capture_qpc}, {"runtime_epoch", response.runtime_epoch}, {"export_generation", response.export_generation}, {"export_sequence", response.export_sequence}, {"producer_result", static_cast<unsigned>(response.result)}, {"source_width", replay.at("defines").at("BUFFER_WIDTH")}, {"source_height", replay.at("defines").at("BUFFER_HEIGHT")}, {"artifacts", nlohmann::json::array()}};
        const char *names[] {"invalid", "source_color", "raw_depth", "candidate", "vertical_majorant", "vertical_field", "final_field", "sbs", "linear_color"};
        for (unsigned i = 0; i < response.texture_count; ++i) {
          const auto &t = response.textures[i];
          const auto kind = static_cast<unsigned>(t.kind);
          check(t.kind == wire::artifact::ui_source_color || (kind > 0 && kind < std::size(names)), "Unknown diagnostic artifact kind");
          const auto *name = t.kind == wire::artifact::ui_source_color ? "ui_source_color" : names[kind];
          const auto file = std::to_string(i) + "_" + name + ".bin";
          std::ofstream stream(output_directory / file, std::ios::binary);
          stream.exceptions(std::ios::failbit | std::ios::badbit);
          stream.write(reinterpret_cast<const char *>(bytes[i].data()), bytes[i].size());
          stream.close();
          manifest["artifacts"].push_back({{"kind", name}, {"artifact_id", kind}, {"file", file}, {"width", t.width}, {"height", t.height}, {"dxgi_format", t.dxgi_format}, {"row_bytes", bytes[i].size() / t.height}, {"byte_count", bytes[i].size()}, {"layout", "tightly packed rows, top to bottom, native little-endian DXGI pixels"}});
        }
        std::ofstream stream(output_directory / "manifest.json", std::ios::binary);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        stream << manifest.dump(2) << '\n';
        stream.close();
      }
    }
  };
}  // namespace sunshine_game3d_test

// Included after the unchanged authored pending-colour fixture helpers. Pure mechanics:
// actual GPU full-source snapshot -> native ROI band/receipt -> frozen7227 retirement.
// No model, positive UI, EMPTY, live-capture, concurrency, or timing authority.
namespace {
  constexpr UINT roi_full_width = 256u, roi_full_height = 128u;
  constexpr UINT roi_left = 16u, roi_top = 8u, roi_width = 128u, roi_height = 64u;

  pattern_owner_t nonzero_roi_owner(live_v2_warp_fixture_t &f) {
    auto owner = pattern_owner(f, 1u);
    auto &r = owner.warm.input_region;
    r.source_width = roi_full_width;
    r.source_height = roi_full_height;
    r.left = roi_left;
    r.top = roi_top;
    r.right = roi_left + roi_width;
    r.bottom = roi_top + roi_height;
    owner.pending.input_region = r;
    pc_require(r.valid(), "Authored nonzero ROI is not a valid own input region");
    // The model CB/OCR/SCT still describe OWN ROI128x64, native bottom band128x32,
    // frame1/clock1001. Physical full256x128 does not replace those old fields.
    return owner;
  }

  cs::constants_t nonzero_roi_current(std::uint64_t serial, std::uint64_t clock) {
    auto c = pattern_current(serial, clock);
    c.source_rect = {roi_left, roi_top, roi_width, roi_height};
    return c;
  }

  std::vector<rgba32f_t> full_roi_pattern(UINT width, UINT height, UINT kind) {
    std::vector<rgba32f_t> out(std::size_t(width) * height);
    for (UINT y = 0; y < height; ++y) {
      for (UINT x = 0; x < width; ++x) {
        out[std::size_t(y) * width + x] = {float(x) / float(width), float(y) / float(height), .125f, 1.f};
      }
    }
    for (UINT y = 0; y < 32u; ++y) {
      for (UINT x = 0; x < 128u; ++x) {
        const UINT px = roi_left + x, py = roi_top + 32u + y;
        if (px >= width || py >= height) {
          continue;
        }
        float v = (x % 4u < 2u) ? 1.f : 0.f;
        if (kind == 1u && x < 48u) {
          v = .5f;
        }
        if (kind == 2u) {
          v = .5f;
        }
        out[std::size_t(py) * width + px] = {v, v, v, 1.f};
      }
    }
    return out;
  }

  std::vector<rgba16_bits_t> read_roi_half(live_v2_warp_fixture_t &f, ID3D11Texture2D *source) {
    D3D11_TEXTURE2D_DESC d {};
    source->GetDesc(&d);
    pc_require(d.Format == DXGI_FORMAT_R16G16B16A16_FLOAT && d.Width == 128u && d.Height == 32u && d.MipLevels == 1u && d.ArraySize == 1u && d.SampleDesc.Count == 1u, "Unexpected actual ROI band descriptor");
    d.Usage = D3D11_USAGE_STAGING;
    d.BindFlags = d.MiscFlags = 0u;
    d.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> stage;
    pc_hr(f.device->CreateTexture2D(&d, nullptr, &stage));
    f.context->CopyResource(stage.Get(), source);
    D3D11_MAPPED_SUBRESOURCE map {};
    pc_hr(f.context->Map(stage.Get(), 0u, D3D11_MAP_READ, 0u, &map));
    std::vector<rgba16_bits_t> out(128u * 32u);
    for (UINT y = 0; y < 32u; ++y) {
      std::memcpy(out.data() + 128u * y, static_cast<const std::byte *>(map.pData) + std::size_t(y) * map.RowPitch, 128u * sizeof(rgba16_bits_t));
    }
    f.context->Unmap(stage.Get(), 0u);
    return out;
  }

  struct roi_source_runtime_t {
    cs::resources_t snapshot;
    std::array<ComPtr<ID3D11ComputeShader>, 3> shaders;
    ComPtr<ID3D11Texture2D> current_texture;
    ComPtr<ID3D11ShaderResourceView> current_srv;

    void prepare(live_v2_warp_fixture_t &f) {
      std::string error;
      pc_require(snapshot.ensure(f.device.Get(), 128u, 32u, error), error.c_str());
      const std::array<models::host_sbs_shader_cache::shader_spec, 3> specs {{{"host_sbs_current_source_snapshot_cs.hlsl", "clear_main", "cs_5_0"}, {"host_sbs_current_source_snapshot_cs.hlsl", "snapshot_main", "cs_5_0"}, {"host_sbs_current_source_snapshot_cs.hlsl", "resolve_main", "cs_5_0"}}};
      const auto source = models::host_sbs_shader_cache::snapshot_sources(pending_root(), specs);
      pc_require(models::host_sbs_shader_cache::source_closure_sha256(source) == "0dca5e8bc18c1795fadaf93a9fbd4b7a6d3d665ec310708f6b78b18235c78244", "Actual snapshot source closure changed");
      for (UINT i = 0; i < 3u; ++i) {
        const auto code = models::host_sbs_shader_cache::get(source, specs[i]);
        pc_require(bool(code), "Actual snapshot bytecode missing");
        pc_hr(f.device->CreateComputeShader(code->data(), code->size(), nullptr, &shaders[i]));
      }
    }

    void dispatch(live_v2_warp_fixture_t &f, const cs::constants_t &current, UINT width, UINT height, UINT kind) {
      const auto pixels = full_roi_pattern(width, height, kind);
      current_texture.Reset();
      current_srv.Reset();
      pc_require(f.create_immutable_texture_srv(DXGI_FORMAT_R32G32B32A32_FLOAT, width, height, sizeof(rgba32f_t), pixels.data(), current_texture, current_srv), "Actual full current texture allocation failed");
      std::array<ID3D11ComputeShader *, 3> raw {};
      for (UINT i = 0; i < 3u; ++i) {
        raw[i] = shaders[i].Get();
      }
      std::string error;
      pc_require(snapshot.dispatch(f.context.Get(), raw, current_srv.Get(), current, error), error.c_str());
    }

    std::vector<std::uint32_t> observe(live_v2_warp_fixture_t &f, pattern_runtime_t &rt, UINT bank, const pattern_owner_t &owner, const cs::constants_t &current) {
      const auto v = snapshot.working_views();
      std::string error;
      // No Map/readback/CPU-generated receipt between the real snapshot and observe.
      pc_require(rt.helper.observe(bank, f.context.Get(), v.source.Get(), v.receipt.Get(), snapshot.working_constants_buffer().Get(), nr::make_expected(current, owner.pending.planned_owner), error), error.c_str());
      return read_words(f.device.Get(), f.context.Get(), rt.helper.export_resources(bank)[12].Get());
    }

    void expect_receipt(live_v2_warp_fixture_t &f, const cs::constants_t &current, UINT width, UINT height, bool ready) {
      const auto v = snapshot.working_views();
      const auto r = read_words(f.device.Get(), f.context.Get(), v.receipt_buffer.Get());
      ASSERT_EQ(r.size(), 64u);
      EXPECT_EQ(r[0], cs::tag);
      EXPECT_EQ(r[1], 1u);
      EXPECT_EQ(r[2], ready ? 1u : 0u);
      EXPECT_EQ(r[40], width);
      EXPECT_EQ(r[41], height);
      EXPECT_EQ(r[42], 128u);
      EXPECT_EQ(r[43], 32u);
      for (UINT i = 0; i < 32u; ++i) {
        UINT value = 0;
        std::memcpy(&value, reinterpret_cast<const std::byte *>(&current) + 4u * i, 4u);
        EXPECT_EQ(r[8u + i], value);
      }
      const auto upload = read_words(f.device.Get(), f.context.Get(), snapshot.working_constants_buffer().Get());
      ASSERT_EQ(upload.size(), 32u);
      EXPECT_EQ(std::memcmp(upload.data(), &current, sizeof(current)), 0);
      if (ready) {
        EXPECT_EQ(r[3], 0u);
        EXPECT_EQ(r[6], 128u * 32u);
        EXPECT_EQ(r[44], 0u);
        EXPECT_EQ(r[45], 1u);
      } else {
        EXPECT_NE(r[3], 0u);
        EXPECT_EQ(r[45], 0u);
      }
      for (UINT i = 46u; i < 64u; ++i) {
        EXPECT_EQ(r[i], 0u);
      }
    }
  };
}  // namespace

TEST(PendingCurrentColourRoiJoinGpu, ActualSnapshotAndRetirementPreservePhysicalRoiAndSeparateClocks) {
  if (pending_root().empty()) {
    GTEST_SKIP() << "Explicit versioned private source root required";
  }
  live_v2_warp_fixture_t f;
  std::string error;
  ASSERT_TRUE(f.initialize(error, warp_shader_e::live_main, false, selected_hardware())) << error;
  const auto owner = nonzero_roi_owner(f);
  pattern_runtime_t rt;
  rt.prepare(f, owner.warm);
  roi_source_runtime_t current;
  current.prepare(f);
  auto c = nonzero_roi_current(2u, 3000u);
  const auto bank = rt.capture(f, owner, c, error);
  ASSERT_TRUE(bank) << error;
  ASSERT_NO_FATAL_FAILURE(expect_authored_owner(f, rt.helper, *bank));
  const auto archived = rt.helper.export_resources(*bank);
  const auto old_cb = read_words(f.device.Get(), f.context.Get(), archived[5].Get());
  const auto old_ocr = read_words(f.device.Get(), f.context.Get(), archived[2].Get());
  const auto old_sct = read_words(f.device.Get(), f.context.Get(), archived[4].Get());
  EXPECT_EQ(old_cb[0], 128u);
  EXPECT_EQ(old_cb[1], 64u);
  EXPECT_EQ(old_cb[12], 1u);
  EXPECT_EQ(old_cb[16], 1001u);
  current.dispatch(f, c, roi_full_width, roi_full_height, 0u);
  const auto same = current.observe(f, rt, *bank, owner, c);
  ASSERT_NO_FATAL_FAILURE(current.expect_receipt(f, c, roi_full_width, roi_full_height, true));
  EXPECT_EQ(same[2], 1u);
  EXPECT_EQ(same[3], 0u);
  EXPECT_EQ(same[4], 256u);
  EXPECT_EQ(same[5], 256u);
  EXPECT_EQ(read_roi_half(f, current.snapshot.working_views().texture.Get()), stripe_band());
  c = nonzero_roi_current(3u, 4000u);
  current.dispatch(f, c, roi_full_width, roi_full_height, 1u);
  const auto partial = current.observe(f, rt, *bank, owner, c);
  ASSERT_NO_FATAL_FAILURE(current.expect_receipt(f, c, roi_full_width, roi_full_height, true));
  EXPECT_EQ(partial[2], 1u);
  EXPECT_EQ(partial[3], 0u);
  EXPECT_EQ(partial[5], 168u);
  c = nonzero_roi_current(4u, 5000u);
  current.dispatch(f, c, roi_full_width, roi_full_height, 2u);
  const auto gone = current.observe(f, rt, *bank, owner, c);
  ASSERT_NO_FATAL_FAILURE(current.expect_receipt(f, c, roi_full_width, roi_full_height, true));
  EXPECT_EQ(gone[2], 2u);
  EXPECT_EQ(gone[3], 0u);
  EXPECT_EQ(gone[4], 256u);
  EXPECT_EQ(gone[5], 0u);
  const auto expected = nr::make_expected(c, owner.pending.planned_owner);
  EXPECT_EQ(std::memcmp(gone.data() + 8u, &expected, sizeof(expected)), 0);
  EXPECT_EQ(read_words(f.device.Get(), f.context.Get(), archived[5].Get()), old_cb);
  EXPECT_EQ(read_words(f.device.Get(), f.context.Get(), archived[2].Get()), old_ocr);
  EXPECT_EQ(read_words(f.device.Get(), f.context.Get(), archived[4].Get()), old_sct);
  EXPECT_EQ(read_roi_half(f, static_cast<ID3D11Texture2D *>(archived[0].Get())), stripe_band());
  RecordProperty("actual_GPU_snapshot_receipt_not_preuploaded", true);
  RecordProperty("old_model1_clock1001_current_material4_clock5000_physical256x128_ROI16_8_128_64", true);
  RecordProperty("negative_only_no_current_glyph_or_EMPTY_authority", true);
}

TEST(PendingCurrentColourRoiJoinGpu, InvalidPhysicalExtentAndScopeCannotUseLatchedRetirement) {
  if (pending_root().empty()) {
    GTEST_SKIP() << "Explicit versioned private source root required";
  }
  live_v2_warp_fixture_t f;
  std::string error;
  ASSERT_TRUE(f.initialize(error, warp_shader_e::live_main, false, selected_hardware())) << error;
  const auto owner = nonzero_roi_owner(f);
  pattern_runtime_t rt;
  rt.prepare(f, owner.warm);
  roi_source_runtime_t current;
  current.prepare(f);
  auto c = nonzero_roi_current(2u, 3000u);
  const auto bank = rt.capture(f, owner, c, error);
  ASSERT_TRUE(bank) << error;
  current.dispatch(f, c, roi_full_width, roi_full_height, 2u);
  const auto retired = current.observe(f, rt, *bank, owner, c);
  ASSERT_EQ(retired[2], 2u);
  ASSERT_EQ(retired[3], 0u);
  const auto owned = rt.helper.export_resources(*bank);
  const auto mark = read_words(f.device.Get(), f.context.Get(), owned[14].Get());
  // Actual smaller full input cannot contain the unchanged ROI. Real snapshot clears native
  // data and publishes context fault; the old group's retained latch is not draw permission.
  c = nonzero_roi_current(3u, 4000u);
  current.dispatch(f, c, 128u, roi_full_height, 2u);
  const auto outside = current.observe(f, rt, *bank, owner, c);
  ASSERT_NO_FATAL_FAILURE(current.expect_receipt(f, c, 128u, roi_full_height, false));
  EXPECT_EQ(outside[2], 0u);
  EXPECT_NE(outside[3], 0u);
  EXPECT_EQ(read_words(f.device.Get(), f.context.Get(), owned[14].Get()), mark);
  const auto cleared = read_roi_half(f, current.snapshot.working_views().texture.Get());
  for (const auto &p : cleared) {
    EXPECT_EQ(p, rgba16_bits_t {});
  }
  // A literal packet corruption test starts from another genuine GPU snapshot. No CPU packet
  // can make an out-of-containment physical extent valid, even if its READY flag is forged.
  current.dispatch(f, c, roi_full_width, roi_full_height, 2u);
  auto damaged = read_words(f.device.Get(), f.context.Get(), current.snapshot.working_views().receipt_buffer.Get());
  damaged[40] = roi_left + roi_width - 1u;
  const auto bad_receipt = words(f.device.Get(), damaged);
  const auto v = current.snapshot.working_views();
  ASSERT_TRUE(rt.helper.observe(*bank, f.context.Get(), v.source.Get(), bad_receipt.srv.Get(), current.snapshot.working_constants_buffer().Get(), nr::make_expected(c, owner.pending.planned_owner), error)) << error;
  const auto corrupt = read_words(f.device.Get(), f.context.Get(), owned[12].Get());
  EXPECT_EQ(corrupt[2], 0u);
  EXPECT_NE(corrupt[3], 0u);
  EXPECT_EQ(read_words(f.device.Get(), f.context.Get(), owned[14].Get()), mark);
  c.scope[2] ^= 1u;
  current.dispatch(f, c, roi_full_width, roi_full_height, 2u);
  const auto wrong_scope = current.observe(f, rt, *bank, owner, c);
  EXPECT_EQ(wrong_scope[2], 0u);
  EXPECT_NE(wrong_scope[3], 0u);
  EXPECT_EQ(read_words(f.device.Get(), f.context.Get(), owned[14].Get()), mark);
  const auto state = read_words(f.device.Get(), f.context.Get(), owned[8].Get());
  EXPECT_EQ(state[9], 1u) << "Invalid transport must not clear or rearm the owner latch";
  RecordProperty("malformed_READY_packet_is_fault_injection_not_source_proof", true);
  RecordProperty("unknown_is_not_retired_or_EMPTY", true);
}

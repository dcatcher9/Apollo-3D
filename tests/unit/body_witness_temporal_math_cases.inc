// Included inside the main fixture's anonymous namespace. These are GRAPH/TEMPORAL
// MECHANICS cases with authored binary graph operands, not full-colour quality claims.
// The separate SavedReplay test executes actual palette/emit/CC once on all raw operands.
class HostSbsBodyWitnessTemporalMathGpuTest: public HostSbsBodyWitnessTemporalGpuTest {
protected:
  struct math_mode_t {
    std::vector<UINT> eligible, seeds;
  };

  std::vector<UINT> math_records_;

  math_mode_t mode() const {
    return {std::vector<UINT>(std::size_t(width_) * crop_height_, 0u), std::vector<UINT>(std::size_t(width_) * crop_height_, 0u)};
  }

  void point(math_mode_t &m, UINT x, UINT y, bool seed = true) {
    m.eligible[pixel(x, y)] = 1u;
    if (seed) {
      m.seeds[pixel(x, y)] = 1u;
    }
  }

  void line(math_mode_t &m, UINT x0, UINT x1, UINT y, UINT seedX) {
    for (UINT x = x0; x < x1; ++x) {
      point(m, x, y, false);
    }
    m.seeds[pixel(seedX, y)] = 1u;
  }

  void rgb_at(UINT x, UINT y, float v) {
    source_[pixel(x, y)] = {colour_half(v), colour_half(v), colour_half(v), colour_half(1.f)};
  }

  void four_black(UINT x, UINT y) {
    for (const auto &o : std::array<std::array<int, 2>, 4> {{{{-2, 0}}, {{2, 0}}, {{0, -2}}, {{0, 2}}}}) {
      const int xx = int(x) + o[0], yy = int(y) + o[1];
      if (xx >= 0 && yy >= 0 && xx < int(width_) && yy < int(crop_height_)) {
        rgb_at(UINT(xx), UINT(yy), 0.f);
      }
    }
  }

  void math_source(UINT modelFrame, UINT sourceSerial) {
    author(modelFrame, true, .15f);
    // Valid authored current OCR geometry; exact same rectangle is used by math palette/domain.
    const auto r = field_rect({4u, 20u, width_ - 4u, crop_height_ - 4u});
    for (UINT offset : {16u, 144u}) {
      std::copy(r.begin(), r.end(), ocr_.begin() + offset);
      ocr_[offset + 4u] = std::bit_cast<UINT>(.9f);
      ocr_[offset + 6u] = 1u;
    }
    frame_for_source(sourceSerial);
  }

  bool run_math_frame(const std::vector<math_mode_t> &modes) {
    if (modes.size() > 8u) {
      return false;
    }
    unbind();
    context_->UpdateSubresource(constants_buffer_.Get(), 0u, nullptr, &constants_, 0u, 0u);
    context_->UpdateSubresource(source_texture_.value.Get(), 0u, nullptr, source_.data(), width_ * 8u, 0u);
    context_->UpdateSubresource(support_texture_.value.Get(), 0u, nullptr, support_.data(), width_ * 4u, 0u);
    context_->UpdateSubresource(p_buffer_.value.Get(), 0u, nullptr, probability_.data(), 0u, 0u);
    context_->UpdateSubresource(ocr_buffer_.value.Get(), 0u, nullptr, ocr_.data(), 0u, 0u);
    const std::array<UINT, 4u> graphShape {width_, crop_height_, width_ * crop_height_, 0u};
    context_->UpdateSubresource(component_constants_.Get(), 0u, nullptr, graphShape.data(), 0u, 0u);
    constexpr std::array<UINT, 4u> zero {};
    context_->ClearUnorderedAccessViewUint(validation_buffer_.uav.Get(), zero.data());
    const colour_request_t validationRequest {{width_, crop_height_, 0u, 0u}};
    context_->UpdateSubresource(request_buffer_.Get(), 0u, nullptr, &validationRequest, 0u, 0u);
    ID3D11Buffer *vcs[] {constants_buffer_.Get(), request_buffer_.Get()};
    context_->CSSetConstantBuffers(0u, 2u, vcs);
    const auto reads = inner_reads();
    context_->CSSetShaderResources(0u, 4u, reads.data());
    ID3D11UnorderedAccessView *validation = validation_buffer_.uav.Get();
    context_->CSSetUnorderedAccessViews(0u, 1u, &validation, nullptr);
    context_->CSSetShader(inner_shaders_[4u].Get(), nullptr, 0u);
    context_->Dispatch((width_ + 15u) / 16u, (crop_height_ + 15u) / 16u, 1u);
    unbind();
    std::array<ID3D11ComputeShader *, 19u> temporalShaders {};
    for (UINT i = 0u; i < 19u; ++i) {
      temporalShaders[i] = temporal_shaders_[i].Get();
    }
    const std::array<UINT, 3u> root {1u, 1u, 1u};
    context_->UpdateSubresource(root_arguments_.Get(), 0u, nullptr, root.data(), 0u, 0u);
    if (!backend_.begin(context_.Get(), temporalShaders, frame_, constants_buffer_.Get(), source_texture_.srv.Get(), validation_buffer_.srv.Get(), root_arguments_.Get(), 0u, error_)) {
      return false;
    }
    std::vector<std::pair<colour_archive_t, colour_archive_t>> baseline;
    const UINT n = width_ * crop_height_;
    std::vector<UINT> empty(n, 0u), domain(n, 0u), cover(n, 0u);
    for (UINT y = 20u; y < crop_height_ - 4u; ++y) {
      for (UINT x = 4u; x < width_ - 4u; ++x) {
        domain[pixel(x, y)] = cover[pixel(x, y)] = 1u;
      }
    }
    const auto rect = field_rect({4u, 20u, width_ - 4u, crop_height_ - 4u});
    for (UINT slot = 0u; slot < 8u; ++slot) {
      unbind();
      std::array<UINT, 128u> palette {};
      palette[0] = 0x35434c49u;
      palette[1] = 1u;
      palette[2] = 1u;
      palette[3] = slot;
      palette[4] = slot == 0u ? UINT(modes.size()) : 0u;
      palette[7] = 1u;
      palette[22] = slot == 0u ? 2u : 3u;
      std::copy(constants_.words.begin(), constants_.words.end(), palette.begin() + 24u);
      for (UINT base : {8u, 12u, 16u}) {
        std::copy(rect.begin(), rect.end(), palette.begin() + base);
      }
      for (UINT i = 0u; i < modes.size(); ++i) {
        const UINT q = 65535u - 1000u * i;
        for (UINT c = 0u; c < 4u; ++c) {
          palette[64u + 8u * i + c] = q;
        }
      }
      context_->UpdateSubresource(palette_buffer_.value.Get(), 0u, nullptr, palette.data(), 0u, 0u);
      if (!backend_.unit_palette(context_.Get(), slot, palette_buffer_.srv.Get(), error_)) {
        return false;
      }
      for (UINT k = 0u; k < 8u; ++k) {
        unbind();
        const bool active = slot == 0u && k < modes.size();
        const auto &eligible = active ? modes[k].eligible : empty;
        const auto &seeds = active ? modes[k].seeds : empty;
        if (eligible.size() != n || seeds.size() != n) {
          return false;
        }
        context_->UpdateSubresource(eligible_texture_.value.Get(), 0u, nullptr, eligible.data(), width_ * 4u, 0u);
        context_->UpdateSubresource(seed_texture_.value.Get(), 0u, nullptr, seeds.data(), width_ * 4u, 0u);
        context_->UpdateSubresource(domain_texture_.value.Get(), 0u, nullptr, domain.data(), width_ * 4u, 0u);
        context_->UpdateSubresource(cover_texture_.value.Get(), 0u, nullptr, cover.data(), width_ * 4u, 0u);
        const std::array<UINT, 3u> args {active ? (width_ + 15u) / 16u : 0u, active ? (crop_height_ + 15u) / 16u : 0u, active ? 1u : 0u};
        context_->UpdateSubresource(poison_arguments_.Get(), 0u, nullptr, args.data(), 0u, 0u);
        const std::array<ID3D11ShaderResourceView *, 4u> graphReads {eligible_texture_.srv.Get(), seed_texture_.srv.Get(), domain_texture_.srv.Get(), cover_texture_.srv.Get()};
        const std::array<ID3D11UnorderedAccessView *, 4u> graphWrites {parent_buffer_.uav.Get(), flags_buffer_.uav.Get(), body_texture_.uav.Get(), final_texture_.uav.Get()};
        context_->CSSetShaderResources(0u, 4u, graphReads.data());
        context_->CSSetUnorderedAccessViews(0u, 4u, graphWrites.data(), nullptr);
        context_->CSSetConstantBuffers(0u, 1u, component_constants_.GetAddressOf());
        for (UINT phase : {0u, 1u, 2u, 3u}) {
          context_->CSSetShader(component_shaders_[phase].Get(), nullptr, 0u);
          context_->DispatchIndirect(poison_arguments_.Get(), 0u);
        }
        unbind();
        if (!backend_.mode_after_ccresolve(context_.Get(), slot, k, seed_texture_.srv.Get(), body_texture_.srv.Get(), parent_buffer_.srv.Get(), flags_buffer_.srv.Get(), palette_buffer_.srv.Get(), cover_texture_.srv.Get(), poison_arguments_.Get(), 0u, error_)) {
          return false;
        }
        if (active) {
          unbind();
          context_->CSSetShaderResources(0u, 4u, graphReads.data());
          context_->CSSetUnorderedAccessViews(0u, 4u, graphWrites.data(), nullptr);
          context_->CSSetConstantBuffers(0u, 1u, component_constants_.GetAddressOf());
          context_->CSSetShader(component_shaders_[5u].Get(), nullptr, 0u);
          context_->DispatchIndirect(poison_arguments_.Get(), 0u);
          unbind();
          baseline.emplace_back(archive(body_texture_.value.Get()), archive(final_texture_.value.Get()));
        }
      }
    }
    if (!backend_.end(context_.Get(), error_)) {
      return false;
    }
    unbind();
    const auto v = backend_.mutable_views();
    ComPtr<ID3D11Resource> b, m, r, records;
    v.body->GetResource(&b);
    v.matte->GetResource(&m);
    v.receipt->GetResource(&r);
    v.records->GetResource(&records);
    ComPtr<ID3D11Texture2D> bt, mt;
    ComPtr<ID3D11Buffer> rb, recordBuffer;
    b.As(&bt);
    m.As(&mt);
    r.As(&rb);
    records.As(&recordBuffer);
    const auto bp = archive(bt.Get()), mp = archive(mt.Get()), rp = archive(rb.Get()), cp = archive(recordBuffer.Get());
    // All GPU work/transport is enqueued BEFORE these Maps. CPU OR below is verifier-only.
    body_.assign(n, 0u);
    final_.assign(n, 0u);
    for (const auto &[x, y] : baseline) {
      const auto xb = colour_words(read_archive(x)), yb = colour_words(read_archive(y));
      for (UINT p = 0u; p < n; ++p) {
        body_[p] |= xb[p];
        final_[p] |= yb[p];
      }
    }
    temporal_body_ = colour_words(read_archive(bp));
    temporal_final_ = colour_words(read_archive(mp));
    temporal_receipt_ = colour_words(read_archive(rp));
    math_records_ = colour_words(read_archive(cp));
    return temporal_receipt_.size() == 256u;
  }

  std::vector<UINT> record_for_root(UINT x, UINT y, UINT modeOrdinal = UINT(-1)) const {
    const UINT bank = temporal_receipt_.at(16u), stride = 512u + 64u * capacities_.components + capacities_.hash_words, base = bank * stride + 512u;
    for (UINT c = 0u; c < temporal_receipt_.at(10u); ++c) {
      const UINT a = base + 64u * c;
      if (math_records_.at(a + 2u) == pixel(x, y) + 1u && (modeOrdinal == UINT(-1) || math_records_.at(a + 1u) == modeOrdinal)) {
        return std::vector<UINT>(math_records_.begin() + a, math_records_.begin() + a + 64u);
      }
    }
    return {};
  }
};

TEST_F(HostSbsBodyWitnessTemporalMathGpuTest, TransitiveOverlapChainAcrossModesReleasesBothCandidates) {
  ASSERT_TRUE(resize(160u, 128u, 0u, 128u));
  for (UINT phase = 1u; phase <= 2u; ++phase) {
    math_source(phase == 1u ? 1u : 3u, phase);
    for (UINT x : {21u, 23u, 25u}) {
      four_black(x, 40u);
    }
    for (UINT x = 20u; x < 27u; ++x) {
      rgb_at(x, 40u, phase == 1u && x >= 25u ? .5f : 1.f);
    }
    rgb_at(25u, 42u, phase == 1u ? .3f : 0.f);
    std::vector<math_mode_t> modes(3u, mode());
    if (phase == 2u) {
      line(modes[0], 20u, 23u, 40u, 21u);
      line(modes[1], 22u, 25u, 40u, 23u);
      line(modes[2], 24u, 27u, 40u, 25u);
    }
    ASSERT_TRUE(run_math_frame(modes)) << error_;
    expect_available();
  }
  EXPECT_EQ(temporal_receipt_[10u], 3u);
  EXPECT_EQ(temporal_receipt_[13u], 0u);
  EXPECT_EQ(temporal_body_, body_);
  EXPECT_EQ(std::count(temporal_body_.begin(), temporal_body_.end(), 1u), 7);
  const auto a = record_for_root(20u, 40u), b = record_for_root(22u, 40u);
  ASSERT_EQ(a.size(), 64u);
  ASSERT_EQ(b.size(), 64u);
  EXPECT_EQ(a[39u], 1u);
  EXPECT_EQ(b[39u], 1u);
  EXPECT_EQ(a[40u], 0u);
  EXPECT_EQ(b[40u], 0u);
  EXPECT_EQ(a[43u], 8u);
  EXPECT_EQ(b[43u], 8u);
}

TEST_F(HostSbsBodyWitnessTemporalMathGpuTest, FourthOriginalBlackWitnessAloneSuppliesOtherComponentEvent) {
  ASSERT_TRUE(resize(160u, 128u, 0u, 128u));
  for (UINT phase = 1u; phase <= 2u; ++phase) {
    math_source(phase == 1u ? 1u : 3u, phase);
    four_black(21u, 40u);
    four_black(41u, 40u);
    for (UINT x = 20u; x < 23u; ++x) {
      rgb_at(x, 40u, phase == 1u ? .5f : 1.f);
    }
    for (UINT x = 40u; x < 43u; ++x) {
      rgb_at(x, 40u, 1.f);
    }
    rgb_at(21u, 42u, phase == 1u ? .3f : 0.f);
    std::vector<math_mode_t> modes(1u, mode());
    if (phase == 2u) {
      line(modes[0], 20u, 23u, 40u, 21u);
      line(modes[0], 40u, 43u, 40u, 41u);
    }
    ASSERT_TRUE(run_math_frame(modes)) << error_;
    expect_available();
  }
  const auto event = record_for_root(20u, 40u);
  ASSERT_EQ(event.size(), 64u);
  EXPECT_EQ(event[32u], 4u);
  EXPECT_EQ(event[38u], 1u);
  EXPECT_EQ(temporal_receipt_[13u], 1u);
  for (UINT x = 40u; x < 43u; ++x) {
    EXPECT_EQ(temporal_body_[pixel(x, 40u)], 0u);
  }
  for (UINT x = 20u; x < 23u; ++x) {
    EXPECT_EQ(temporal_body_[pixel(x, 40u)], 1u);
  }
}

TEST_F(HostSbsBodyWitnessTemporalMathGpuTest, EqualWitnessCountDifferentCoordinateSetCannotCarry) {
  ASSERT_TRUE(resize(160u, 128u, 0u, 128u));
  UINT previousWitnessCount = 0u;
  for (UINT phase = 1u; phase <= 3u; ++phase) {
    const UINT model = phase == 1u ? 1u : phase == 2u ? 3u :
                                                        6u;
    math_source(model, phase);
    four_black(21u, 40u);
    four_black(41u, 40u);
    four_black(44u, 40u);
    for (UINT x = 20u; x < 23u; ++x) {
      rgb_at(x, 40u, phase == 1u ? .5f : 1.f);
    }
    for (UINT x = 40u; x < 46u; ++x) {
      rgb_at(x, 40u, 1.f);
    }
    rgb_at(21u, 42u, phase == 1u ? .3f : 0.f);
    std::vector<math_mode_t> modes(1u, mode());
    if (phase > 1u) {
      line(modes[0], 20u, 23u, 40u, 21u);
      line(modes[0], 40u, 46u, 40u, phase == 2u ? 41u : 44u);
    }
    ASSERT_TRUE(run_math_frame(modes)) << error_;
    expect_available();
    if (phase == 2u) {
      const auto c = record_for_root(40u, 40u);
      ASSERT_EQ(c.size(), 64u);
      previousWitnessCount = c[32u];
      EXPECT_EQ(temporal_receipt_[13u], 1u);
    }
  }
  const auto c = record_for_root(40u, 40u);
  ASSERT_EQ(c.size(), 64u);
  EXPECT_EQ(c[32u], previousWitnessCount);
  EXPECT_EQ(c[36u], 1u);
  EXPECT_EQ(c[37u], 0u);
  EXPECT_EQ(c[40u], 0u);
  for (UINT x = 40u; x < 46u; ++x) {
    EXPECT_EQ(temporal_body_[pixel(x, 40u)], 1u);
  }
}

TEST_F(HostSbsBodyWitnessTemporalMathGpuTest, CurrentCountAbove512UsesFinalNegativeBudgetAfterClosure) {
  ASSERT_TRUE(resize(160u, 128u, 0u, 128u));
  for (bool releaseOne : {false, true}) {
    backend_ = temporal::resources_t {};
    ASSERT_TRUE(backend_.ensure(device_.Get(), width_, crop_height_, capacities_, error_));
    for (UINT phase = 1u; phase <= 2u; ++phase) {
      math_source(phase == 1u ? 1u : 3u, phase);
      std::vector<std::array<UINT, 2u>> dots;
      for (UINT i = 0u; i < 513u; ++i) {
        dots.push_back({8u + 4u * (i % 32u), 24u + 4u * (i / 32u)});
      }
      for (auto p : dots) {
        four_black(p[0], p[1]);
      }
      if (releaseOne) {
        four_black(9u, 24u);
      } else {
        four_black(148u, 112u);
      }
      for (auto p : dots) {
        rgb_at(p[0], p[1], 1.f);
      }
      if (releaseOne) {
        rgb_at(9u, 24u, phase == 1u ? .5f : 1.f);
        rgb_at(10u, 24u, phase == 1u ? .5f : 1.f);
        rgb_at(9u, 26u, phase == 1u ? .3f : 0.f);
      } else {
        rgb_at(148u, 112u, phase == 1u ? .5f : 1.f);
        rgb_at(148u, 114u, phase == 1u ? .3f : 0.f);
      }
      std::vector<math_mode_t> modes(releaseOne ? 2u : 1u, mode());
      if (phase == 2u) {
        for (auto p : dots) {
          point(modes[0], p[0], p[1]);
        }
        if (releaseOne) {
          line(modes[1], 8u, 11u, 24u, 9u);
        } else {
          point(modes[0], 148u, 112u);
        }
      }
      ASSERT_TRUE(run_math_frame(modes)) << error_;
      expect_available();
    }
    EXPECT_EQ(temporal_receipt_[10u], 514u);
    EXPECT_EQ(temporal_receipt_[15u], releaseOne ? 0u : 1u);
    EXPECT_EQ(temporal_receipt_[13u], releaseOne ? 512u : 0u);
    if (!releaseOne) {
      EXPECT_EQ(temporal_body_, body_);
    } else {
      EXPECT_EQ(std::count(temporal_body_.begin(), temporal_body_.end(), 1u), 3);
      const auto a = record_for_root(8u, 24u, 0u);
      ASSERT_EQ(a.size(), 64u);
      EXPECT_EQ(a[39u], 1u);
      EXPECT_EQ(a[40u], 0u);
      EXPECT_EQ(a[43u], 8u);
    }
  }
}

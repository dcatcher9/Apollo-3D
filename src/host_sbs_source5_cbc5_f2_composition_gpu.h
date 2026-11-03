#pragma once

// Same-current Source5/CBC5/F2 composition. Queue success establishes submission,
// not GPU completion or drawing authority. Original display/P/S/OCR/UI/SCT operands
// are read-only; the selector authenticates the resulting publication separately.
#ifdef _WIN32
#include "src/host_sbs_current_outline_support_closure_gpu.h"
#include "src/host_sbs_display_snapshot_rgb8_analysis_gpu.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace models::source5_cbc5_f2_composition_gpu_v1 {
using Microsoft::WRL::ComPtr;
namespace source5 = display_snapshot_rgb8_analysis_gpu_v1;
namespace f2 = host_sbs_current_outline_support_closure_gpu_v2;
namespace cbc = host_sbs_current_black_enclosure_gpu_v1;
namespace raw = host_sbs_current_black_ring_gpu_v1;

struct program_t {
  source5::program_t source;
  std::array<ID3D11ComputeShader *, 5u> colour{};
  std::array<ID3D11ComputeShader *, 6u> components{};
  // Only shader arrays are consumed. Caller context/producer/snapshot fields
  // are ignored.
  f2::program_t outline;
};

struct request_t {
  source5::request_t source;
  ID3D11ShaderResourceView *probability = nullptr;
  ID3D11ShaderResourceView *support = nullptr;
  ID3D11ShaderResourceView *ocr_record = nullptr;
  ID3D11ShaderResourceView *actual_uip40 = nullptr;
  ID3D11ShaderResourceView *original_legacy_matte = nullptr;
  UINT postprocess_mode =
      0u; // Actual pending function result1/2/4, never guessed from root80.
  UINT expected_work =
      0u; // Actual pending work0/1/2. Never relabel non-OOCR work as1.
};

struct current_operands_t {
  ComPtr<ID3D11DeviceContext> ordered_context;
  ComPtr<ID3D11ShaderResourceView> legacy_display_snapshot, parent_raw_source,
      sct;
  ComPtr<ID3D11ShaderResourceView> probability, support, ocr_record,
      actual_uip40, original_legacy_matte;
  ComPtr<ID3D11Buffer> actual_contour_constants96, actual_final256;
  std::array<UINT, 4u> parent_roi_ltrb{};
  std::uint64_t frame = 0u, observation_us = 0u, analysis_generation = 0u,
                token = 0u;
  UINT parent_transfer = 0u, postprocess_mode = 0u, expected_work = 0u;

  [[nodiscard]] source5::request_t source_request() const {
    source5::request_t r;
    r.display_snapshot_band = legacy_display_snapshot.Get();
    r.parent_raw_source = parent_raw_source.Get();
    r.actual_contour_constants96 = actual_contour_constants96.Get();
    r.actual_sct32 = sct.Get();
    r.actual_final256 = actual_final256.Get();
    r.roi_ltrb = parent_roi_ltrb;
    r.frame = frame;
    r.observation_us = observation_us;
    r.analysis_generation = analysis_generation;
    r.token = token;
    r.transfer_mode = parent_transfer;
    return r;
  }
};

struct bundle_t {
  std::uint64_t host_sequence = 0u, owner_epoch = 0u;
  // True ONLY means all three helper queue calls returned commands-submitted
  // success. It is not GPU completion/availability, token proof, EMPTY, birth,
  // DDup or draw authority.
  bool all_stages_issued = false;
  UINT width = 0u, full_height = 0u, crop_top = 0u, band_height = 0u;
  current_operands_t current;
  source5::views_t source;
  cbc::mutable_views_t enclosure;
  raw::mutable_views_t ring;
  f2::mutable_views_t outline;
  ComPtr<ID3D11ShaderResourceView> baseline_matte, numeric_proof; // Exact same queued current baseline.
};

class resources_t {
  ComPtr<ID3D11Device> device_;
  source5::resources_t source_;
  f2::resources_t f2_;
  current_operands_t current_;
  std::array<ComPtr<ID3D11ComputeShader>, 52u> shader_lease_;
  // Optional all-or-none cache programs do not change the mandatory52 contract.
  std::array<ComPtr<ID3D11ComputeShader>, 3u> stability_shader_lease_;
  UINT width_ = 0u, full_height_ = 0u, crop_top_ = 0u, band_height_ = 0u;
  std::uint64_t sequence_ = 0u, owner_epoch_ = 0u;
  bool ready_ = false, recorded_ = true, all_stages_issued_ = false;

  bool same_device(ID3D11DeviceChild *child) const {
    if (!child || !device_)
      return false;
    ComPtr<ID3D11Device> d;
    child->GetDevice(&d);
    return d.Get() == device_.Get();
  }

  bool structured_view(ID3D11ShaderResourceView *view, UINT words) const {
    if (!same_device(view))
      return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC s{};
    view->GetDesc(&s);
    ComPtr<ID3D11Resource> r;
    view->GetResource(&r);
    ComPtr<ID3D11Buffer> b;
    if (FAILED(r.As(&b)))
      return false;
    D3D11_BUFFER_DESC d{};
    b->GetDesc(&d);
    return s.ViewDimension == D3D11_SRV_DIMENSION_BUFFER &&
           s.Format == DXGI_FORMAT_UNKNOWN && s.Buffer.FirstElement == 0u &&
           s.Buffer.NumElements == words && d.ByteWidth == words * 4u &&
           d.StructureByteStride == 4u &&
           (d.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED) != 0u;
  }

  bool native_uint_view(ID3D11ShaderResourceView *view) const {
    if (!same_device(view))
      return false;
    D3D11_SHADER_RESOURCE_VIEW_DESC s{};
    view->GetDesc(&s);
    ComPtr<ID3D11Resource> r;
    view->GetResource(&r);
    ComPtr<ID3D11Texture2D> t;
    if (FAILED(r.As(&t)))
      return false;
    D3D11_TEXTURE2D_DESC d{};
    t->GetDesc(&d);
    return s.ViewDimension == D3D11_SRV_DIMENSION_TEXTURE2D &&
           s.Format == DXGI_FORMAT_R32_UINT &&
           s.Texture2D.MostDetailedMip == 0u &&
           (s.Texture2D.MipLevels == 1u || s.Texture2D.MipLevels == ~0u) &&
           d.Width == width_ && d.Height == band_height_ &&
           d.Format == DXGI_FORMAT_R32_UINT && d.MipLevels == 1u &&
           d.ArraySize == 1u && d.SampleDesc.Count == 1u &&
           d.SampleDesc.Quality == 0u;
  }

  bool preflight(ID3D11DeviceContext *context, const program_t &p,
                 const request_t &r,
                 std::array<ComPtr<ID3D11ComputeShader>, 52u> &shaders,
                 std::array<ComPtr<ID3D11ComputeShader>, 3u> &stability_shaders,
                 std::string &error) const {
    if (!ready_ || !same_device(context) ||
        context->GetType() != D3D11_DEVICE_CONTEXT_IMMEDIATE ||
        (r.postprocess_mode != 1u && r.postprocess_mode != 2u &&
         r.postprocess_mode != 4u) ||
        r.expected_work > 2u || !r.source.frame || !r.source.observation_us ||
        !r.source.token || r.source.transfer_mode > 2u) {
      error =
          "Composition requires an actual prepared same-device pending tuple";
      return false;
    }
    for (auto *view :
         {r.source.display_snapshot_band, r.source.parent_raw_source,
          r.source.actual_sct32, r.probability, r.support, r.ocr_record,
          r.actual_uip40, r.original_legacy_matte}) {
      if (!same_device(view)) {
        error = "Composition current operand missing or on another device";
        return false;
      }
    }
    if (!same_device(r.source.actual_contour_constants96) ||
        !same_device(r.source.actual_final256)) {
      error = "Composition current CB/root missing or on another device";
      return false;
    }
    constexpr UINT pwords = depth_coordinate_v2::subtitle_ocr_output_width *
                            depth_coordinate_v2::subtitle_ocr_output_height;
    if (!structured_view(r.probability, pwords) ||
        !structured_view(r.ocr_record, 256u) ||
        !structured_view(r.actual_uip40, 40u) || !native_uint_view(r.support) ||
        !native_uint_view(r.original_legacy_matte)) {
      error = "Composition current P/S/OCR/UIP/legacy shape or format mismatch";
      return false;
    }
    UINT count = 0u;
    auto add = [&](ID3D11ComputeShader *s) {
      if (!same_device(s) || count >= shaders.size())
        return false;
      shaders[count++] = s;
      return true;
    };
    if (!add(p.source.prepare) || !add(p.source.copy) ||
        !add(p.source.finalize) || !add(p.source.envelope)) {
      error = "Composition recipe5 shaders unavailable";
      return false;
    }
    for (auto *s : p.colour)
      if (!add(s)) {
        error = "Composition colour shader unavailable";
        return false;
      }
    for (auto *s : p.components)
      if (!add(s)) {
        error = "Composition component shader unavailable";
        return false;
      }
    for (auto *s : p.outline.enclosure.shaders)
      if (!add(s)) {
        error = "Composition CBC5 shader unavailable";
        return false;
      }
    for (auto *s : p.outline.enclosure.roi_graph)
      if (!add(s)) {
        error = "Composition CBC ROI shader unavailable";
        return false;
      }
    for (auto *s : p.outline.shaders)
      if (!add(s)) {
        error = "Composition raw shader unavailable";
        return false;
      }
    for (auto *s : p.outline.roi_graph)
      if (!add(s)) {
        error = "Composition raw ROI shader unavailable";
        return false;
      }
    for (auto *s : p.outline.band_shaders)
      if (!add(s)) {
        error = "Composition F2 shader unavailable";
        return false;
      }
    for (auto *s : p.outline.inside_shaders)
      if (!add(s)) {
        error = "Composition independent Inside shader unavailable";
        return false;
      }
    if (count != shaders.size()) {
      error = "Composition shader bundle shape mismatch";
      return false;
    }
    bool stability_available = true;
    for (auto *s : p.outline.stability_shaders)
      stability_available = stability_available && same_device(s);
    if (stability_available)
      for (UINT i = 0u; i < stability_shaders.size(); ++i)
        stability_shaders[i] = p.outline.stability_shaders[i];
    return true;
  }

  static current_operands_t retain(ID3D11DeviceContext *context,
                                   const request_t &r) {
    current_operands_t c;
    c.ordered_context = context;
    c.legacy_display_snapshot = r.source.display_snapshot_band;
    c.parent_raw_source = r.source.parent_raw_source;
    c.sct = r.source.actual_sct32;
    c.actual_contour_constants96 = r.source.actual_contour_constants96;
    c.actual_final256 = r.source.actual_final256;
    c.probability = r.probability;
    c.support = r.support;
    c.ocr_record = r.ocr_record;
    c.actual_uip40 = r.actual_uip40;
    c.original_legacy_matte = r.original_legacy_matte;
    c.parent_roi_ltrb = r.source.roi_ltrb;
    c.frame = r.source.frame;
    c.observation_us = r.source.observation_us;
    c.analysis_generation = r.source.analysis_generation;
    c.token = r.source.token;
    c.parent_transfer = r.source.transfer_mode;
    c.postprocess_mode = r.postprocess_mode;
    c.expected_work = r.expected_work;
    return c;
  }

public:
  resources_t() = default;
  resources_t(const resources_t &) = delete;
  resources_t &operator=(const resources_t &) = delete;
  resources_t(resources_t &&) = default;
  resources_t &operator=(resources_t &&) = default;

  void set_recorded_execution(bool enabled) noexcept {
    recorded_ = enabled;
    f2_.set_recorded_execution(enabled);
  }

  [[nodiscard]] f2::diagnostics_t diagnostics() const noexcept {
    return f2_.diagnostics();
  }
  [[nodiscard]] cbc::diagnostics_t parent_diagnostics() const noexcept {
    return f2_.parent_diagnostics();
  }

  bool ensure(ID3D11Device *device, UINT width, UINT full_height, UINT crop_top,
              UINT band_height, std::string &error) {
    error.clear();
    all_stages_issued_ = false;
    if (ready_ && device_.Get() == device && width_ == width &&
        full_height_ == full_height && crop_top_ == crop_top &&
        band_height_ == band_height)
      return true;
    if (owner_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
      error = "Composition owner epoch exhausted";
      return false;
    }
    resources_t next;
    next.recorded_ = recorded_;
    next.f2_.set_recorded_execution(recorded_);
    next.sequence_ = sequence_;
    next.owner_epoch_ = owner_epoch_ + 1u;
    if (!next.source_.ensure(device, width, full_height, crop_top, band_height,
                             error) ||
        !next.f2_.ensure(device, width, band_height, error))
      return false;
    next.device_ = device;
    next.width_ = width;
    next.full_height_ = full_height;
    next.crop_top_ = crop_top;
    next.band_height_ = band_height;
    next.ready_ = true;
    *this = std::move(next);
    return true;
  }

  // Same serialized immediate interval through selector. Caller already owns
  // the actual admitted parent-RAW/color-slot lease; AddRef here cannot freeze
  // or recover an old frame. A false result may follow partial submission:
  // retain this owner until ordered consumption, mark new path unsupplied, and
  // select ORIGINAL LEGACY. Never use an analysis-baseline M.
  bool queue(ID3D11DeviceContext *context, const program_t &p,
             const request_t &r, std::string &error) {
    error.clear();
    all_stages_issued_ = false;
    std::array<ComPtr<ID3D11ComputeShader>, 52u> shaders;
    std::array<ComPtr<ID3D11ComputeShader>, 3u> stability_shaders;
    if (!preflight(context, p, r, shaders, stability_shaders, error))
      return false;
    if (sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      error = "Composition host sequence exhausted";
      return false;
    }
    ++sequence_;
    // Retain BEFORE the first command, including partial-failure paths.
    // Resources remain mutable: caller may not start another queue/ensure/write
    // before selector consumption.
    current_ = retain(context, r);
    shader_lease_ = std::move(shaders);
    stability_shader_lease_ = std::move(stability_shaders);
    const auto source_request = current_.source_request();
    if (!source_.queue(context, p.source, source_request, error))
      return false;
    if (!source_.queue_uip_envelope(context, p.source,
                                    current_.actual_uip40.Get(), source_request,
                                    error))
      return false;
    const auto v = source_.views();
    f2::program_t program;
    program.shaders = p.outline.shaders;
    program.roi_graph = p.outline.roi_graph;
    program.band_shaders = p.outline.band_shaders;
    program.inside_shaders = p.outline.inside_shaders;
    for (UINT i = 0u; i < program.stability_shaders.size(); ++i)
      program.stability_shaders[i] = stability_shader_lease_[i].Get();
    program.enclosure.shaders = p.outline.enclosure.shaders;
    program.enclosure.roi_graph = p.outline.enclosure.roi_graph;
    program.enclosure.current_sct = current_.sct.Get();
    program.enclosure.current_uip = v.uip_envelope.Get();
    auto &producer = program.enclosure.producer;
    producer.original_source = v.reference_rgb8.Get();
    producer.transfer = cbc::transfer_t::encoded_sdr;
    producer.input_known = false;
    producer.direct_encoded_recipe = 5u;
    producer.postprocess_mode = current_.postprocess_mode;
    producer.expected_work = current_.expected_work;
    producer.transaction_token = current_.token;
    producer.model_frame = current_.frame;
    producer.model_observation_us = current_.observation_us;
    producer.analysis_generation = current_.analysis_generation;
    producer.analysis_roi = {0u, 0u, width_, full_height_};
    // Production composition never copies test snapshots or allocates
    // diagnostic planes.
    program.packed_temp_snapshot = nullptr;
    program.passable_cache_snapshot = nullptr;
    if (!f2_.dispatch(context, p.colour, p.components,
                      current_.actual_contour_constants96.Get(),
                      v.analysis_band.Get(), current_.probability.Get(),
                      current_.support.Get(), current_.ocr_record.Get(),
                      current_.actual_final256.Get(), 240u, error, &program))
      return false;
    all_stages_issued_ = true;
    return true;
  }

  // Strong current operands and views, but NOT immutable pixel snapshots. These
  // are valid only for this owner's serialized queue->selector interval and its
  // exact host identity.
  [[nodiscard]] bundle_t bundle() const {
    const auto baseline=f2_.baseline_views();
    return {sequence_,          owner_epoch_,
            all_stages_issued_, width_,
            full_height_,       crop_top_,
            band_height_,       current_,
            source_.views(),    f2_.enclosure_views(),
            f2_.raw_views(),    f2_.mutable_views(),
            baseline.matte,    baseline.validation_receipt};
  }
};
} // namespace models::source5_cbc5_f2_composition_gpu_v1
#endif

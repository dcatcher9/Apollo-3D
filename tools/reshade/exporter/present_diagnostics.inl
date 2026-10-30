// SPDX-License-Identifier: GPL-3.0-only
// Part of exporter.cpp, the same translation unit (tools/reshade/README.md, Source map),
// included at the end of the file: the present diagnostics. The process's video memory
// (query_vram) and a published frame's output sample (classify_output), and publisher_t's
// present timing window (finish_present_timing, close_timing_window), present gaps
// (measure_present_gap, log_present_gap), VRAM adapter (vram_adapter) and SBS output
// line (log_output). The members were in-class bodies of publisher_t and stay inline;
// publisher_t declares them and keeps the present-gap constants and present_gap_t, and
// the shell's part declarations name the two free functions.
namespace {
  // This process's video memory on the adapter (game3d_vram.h): two CPU-only
  // DXGI queries, made at most every vram::sample_period_ms.
  sunshine_game3d::vram::reading query_vram(IDXGIAdapter3 *adapter) {
    using sunshine_game3d::vram::to_mib;
    sunshine_game3d::vram::reading out;
    DXGI_QUERY_VIDEO_MEMORY_INFO local {}, nonlocal {};
    if (!adapter || FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) ||
        FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal))) {
      return out;
    }
    out.local_usage_mib = to_mib(local.CurrentUsage);
    out.local_budget_mib = to_mib(local.Budget);
    out.nonlocal_usage_mib = to_mib(nonlocal.CurrentUsage);
    out.nonlocal_budget_mib = to_mib(nonlocal.Budget);
    out.valid = true;
    return out;
  }

  output_sample classify_output(const frame_decision_t &frame) {
    const auto &scene = frame.scene;
    const bool placed = scene.ready && scene.blend > 0.f;
    output_sample out;
    out.fg = frame.diagnostics.frame_generation_active;
    out.scene_flat = scene.owned && frame.depth_ready && !placed;
    out.scene_fading = scene.owned && frame.depth_ready && placed && scene.blend < 1.f;
    out.depth_ready = frame.depth_ready;
    out.reused_depth = frame.reused_depth;
    out.gap = frame.gap;
    return out;
  }

  inline void publisher_t::finish_present_timing(api::effect_runtime *runtime, const std::array<LARGE_INTEGER, 5> &marks, unsigned count) {
    if (!count) return;
    LARGE_INTEGER end{}, frequency{};
    QueryPerformanceCounter(&end);
    QueryPerformanceFrequency(&frequency);
    const auto ms = [&](const LARGE_INTEGER &from, const LARGE_INTEGER &to) {
      return double(to.QuadPart - from.QuadPart) * 1000.0 / double(frequency.QuadPart);
    };
    const double cpu_ms = ms(marks[0], end);
    // Sections end at the next mark, or at the end when the present left early.
    std::array<double, 5> sections{};
    for (unsigned i = 0; i < count; ++i) sections[i] = ms(marks[i], i + 1 < count ? marks[i + 1] : end);
    const auto now = GetTickCount64();
    char message[1024]{};
    unsigned long long window_ms = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto &proof = runtimes_[runtime];
      auto &timing = proof.timing;
      ++timing.presents; timing.cpu_sum_ms += cpu_ms;
      if (cpu_ms > timing.cpu_max_ms) { timing.cpu_max_ms = cpu_ms; timing.worst = sections; }
      if (!timing.next_log) {
        timing.start = now;
        timing.next_log = now + 10000;
      }
      if (now < timing.next_log) return;
      window_ms = now >= timing.start ? now - timing.start : 0;
      close_timing_window(proof, now, message);
    }
    log(reshade::log::level::info, message);
    // The game's video memory over the same window (game3d_vram.h): its own
    // line, since the timing line's buffer is nearly full.
    char memory[384];
    if (sunshine_game3d::vram::format_window_line(memory, sizeof(memory), sunshine_game3d::vram::process().take_window(),
          window_ms))
      log(reshade::log::level::info, memory);
  }

  // Requires mutex_. Every Present of the runtime counts, Game 3D enabled
  // or not; fg is the frame-generation state of the Present before the gap.
  inline publisher_t::present_gap_t publisher_t::measure_present_gap(runtime_t &proof, std::int64_t qpc, std::uint64_t now) {
    present_gap_t gap;
    if (proof.last_present_qpc && qpc > proof.last_present_qpc) {
      LARGE_INTEGER frequency {};
      QueryPerformanceFrequency(&frequency);
      gap.ms = double(qpc - proof.last_present_qpc) * 1000.0 / double(frequency.QuadPart);
    }
    proof.last_present_qpc = qpc;
    if (gap.ms < present_gap_ms) return gap;
    gap.fg = proof.frame.diagnostics.frame_generation_active;
    if (!proof.gap_window_start || now - proof.gap_window_start >= 10000) {
      proof.gap_window_start = now;
      proof.gap_lines = 0;
    }
    if (proof.gap_lines >= present_gap_lines_per_window) {
      ++proof.gap_suppressed;
      return gap;
    }
    ++proof.gap_lines;
    gap.suppressed = std::exchange(proof.gap_suppressed, 0u);
    gap.log = true;
    return gap;
  }

  // Enabled Presents only: a disabled Present touches no window or device.
  inline void publisher_t::log_present_gap(api::effect_runtime *runtime, const present_gap_t &gap) {
    const sunshine_game3d::vram::latest_suffix memory;
    const bool foreground = owns_foreground(runtime, static_cast<HWND>(runtime->get_hwnd()));
    char message[256];
    std::snprintf(message, sizeof(message),
      "Sunshine Game 3D present gap: %.0f ms since the previous Present; fg=%d foreground=%d suppressed=%u%s", gap.ms,
      int(gap.fg), int(foreground), gap.suppressed, memory.text);
    log(reshade::log::level::info, message);
  }

  // Requires mutex_. The adapter of the runtime's device, opened once (a
  // failure is not retried); the caller adds its own reference to use it
  // outside the lock.
  inline IDXGIAdapter3 *publisher_t::vram_adapter(runtime_t &proof, api::device *device, IDXGISwapChain *swapchain) {
    if (proof.vram_adapter_checked) return proof.vram_adapter.get();
    proof.vram_adapter_checked = true;
    com_ptr<IDXGIAdapter> adapter;
    if (device->get_api() == api::device_api::d3d11) {
      com_ptr<IDXGIDevice> dxgi;
      auto *native = reinterpret_cast<ID3D11Device *>(device->get_native());
      if (FAILED(native->QueryInterface(IID_PPV_ARGS(dxgi.put()))) || FAILED(dxgi->GetAdapter(adapter.put()))) return nullptr;
    } else {
      // A D3D12 device has no DXGI parent: the swapchain's factory opens
      // the device's adapter by its LUID.
      com_ptr<IDXGIFactory4> factory;
      const LUID luid = reinterpret_cast<ID3D12Device *>(device->get_native())->GetAdapterLuid();
      if (!swapchain || FAILED(swapchain->GetParent(IID_PPV_ARGS(factory.put()))) ||
          FAILED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(adapter.put())))) return nullptr;
    }
    if (FAILED(adapter->QueryInterface(IID_PPV_ARGS(proof.vram_adapter.put())))) return nullptr;
    return proof.vram_adapter.get();
  }

  // Requires mutex_. Formats the timing line of the window ending now into
  // message and starts the next window. window_ms is the window's length:
  // 10 s from a periodic line, shorter for the last window of a runtime
  // reset (invalidate), which closes it early so its counts (the UI layer's
  // skipped copies among them) are logged rather than lost or carried into
  // the next runtime's first line.
  inline void publisher_t::close_timing_window(runtime_t &proof, std::uint64_t now, char (&message)[1024]) {
    auto &timing = proof.timing;
    sunshine_game3d::gpu_timing gpu;
    if (proof.renderer) proof.renderer->take_gpu_timing(gpu);
    using stage = sunshine_game3d::gpu_timing;
    // The offscreen UI layer's live copies over the same interval:
    // recorded, skipped by a saturated ring, the offered copy's Present
    // count over the Presents it was offered to, the longest time the
    // layer went unrefreshed across skipped copies (ui_layer::skip_gap),
    // and qualifying clears the tracker dropped with every entry taken
    // (ui_layer::tracker_capacity).
    const auto layer = sunshine_game3d::ui_layer::take_stats();
    std::snprintf(message, sizeof(message),
      "Sunshine Game 3D timing: presents=%llu cpu_ms={mean=%.3f max=%.3f} cpu_worst_ms={setup=%.3f depth=%.3f ui=%.3f render=%.3f export=%.3f} gpu_frames=%u gpu_ms mean/max={total=%.3f/%.3f inputs=%.3f/%.3f source=%.3f/%.3f detection=%.3f/%.3f linearize=%.3f/%.3f candidate=%.3f/%.3f vertical=%.3f/%.3f horizontal=%.3f/%.3f eyes=%.3f/%.3f pack=%.3f/%.3f} gpu_profile=%s dropped_fence_pending=%u dropped_unresolved=%u incomplete=%u ui_layer={copies=%llu skipped=%llu offers=%llu presents_since_copy={mean=%.2f max=%u} skip_gap_ms=%.1f targets_dropped=%llu} window_ms=%llu",
      static_cast<unsigned long long>(timing.presents),
      timing.presents ? timing.cpu_sum_ms / double(timing.presents) : 0.0, timing.cpu_max_ms,
      timing.worst[0], timing.worst[1], timing.worst[2], timing.worst[3], timing.worst[4], gpu.frames,
      gpu.mean_ms[stage::total], gpu.max_ms[stage::total], gpu.mean_ms[stage::inputs], gpu.max_ms[stage::inputs],
      gpu.mean_ms[stage::source], gpu.max_ms[stage::source], gpu.mean_ms[stage::detection], gpu.max_ms[stage::detection],
      gpu.mean_ms[stage::linearize], gpu.max_ms[stage::linearize], gpu.mean_ms[stage::candidate], gpu.max_ms[stage::candidate],
      gpu.mean_ms[stage::vertical], gpu.max_ms[stage::vertical], gpu.mean_ms[stage::horizontal], gpu.max_ms[stage::horizontal],
      gpu.mean_ms[stage::eyes], gpu.max_ms[stage::eyes], gpu.mean_ms[stage::pack], gpu.max_ms[stage::pack],
      sunshine_game3d::name(gpu.state), gpu.dropped_fence_pending, gpu.dropped_unresolved, gpu.incomplete,
      static_cast<unsigned long long>(layer.copies), static_cast<unsigned long long>(layer.skipped),
      static_cast<unsigned long long>(layer.offers),
      layer.offers ? double(layer.presents_since_sum) / double(layer.offers) : 0.0, layer.presents_since_max,
      double(layer.skip_gap_max_us) / 1000.0, static_cast<unsigned long long>(layer.targets_dropped),
      static_cast<unsigned long long>(now >= timing.start ? now - timing.start : 0));
    timing = {};
    timing.start = now;
    timing.next_log = now + 10000;
  }

  inline void publisher_t::log_output(api::effect_runtime *runtime, runtime_t &proof, std::uint64_t now) {
    auto &counts = proof.output;
    if (now < counts.next_log) return;
    char text[896]{};
    std::snprintf(text, sizeof(text), "Sunshine SBS output: published=%llu fg=%llu scene_flat=%llu scene_fading=%llu published_fresh_depth=%llu published_reused_depth=%llu published_depth_missing=%llu (unavailable=%llu reuse_after_gap=%llu reuse_other_source=%llu) runtime=0x%llx generation=%llu dropped=%llu overwritten_unconsumed=%llu; cumulative Game 3D publications; scene_flat had depth but showed the colour frame; reused depth is bounded and does not advance calibration; unavailable reasons are in Sunshine depth readiness; dropped found no reusable export slot, overwritten_unconsumed replaced a ready slot the consumer never claimed",
      static_cast<unsigned long long>(counts.published), static_cast<unsigned long long>(counts.fg),
      static_cast<unsigned long long>(counts.scene_flat), static_cast<unsigned long long>(counts.scene_fading),
      static_cast<unsigned long long>(counts.published_fresh_depth), static_cast<unsigned long long>(counts.published_reused_depth),
      static_cast<unsigned long long>(counts.published_depth_missing),
      static_cast<unsigned long long>(counts.missing_unavailable), static_cast<unsigned long long>(counts.missing_reuse_after_gap),
      static_cast<unsigned long long>(counts.missing_reuse_other_source),
      static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(runtime)),
      static_cast<unsigned long long>(ring_.generation_ ? ring_.generation_->id : 0),
      static_cast<unsigned long long>(counts.dropped), static_cast<unsigned long long>(counts.overwritten_unconsumed));
    log(reshade::log::level::info, text);
    counts.next_log = now + 5000;
  }
}  // namespace
